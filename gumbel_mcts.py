"""
Gumbel MuZero MCTS — policy improvement by planning with Gumbel noise.

Based on: Danihelka et al. 2022, "Policy improvement by planning with Gumbel"
Key insight: Sequential Halving + Gumbel noise gives provably improved policies
even with very few simulations (2-16).

Drop-in replacement for MCTS — exposes the same search() interface.
"""

import math

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from data import fen_to_graph


class MCTSNode:
    """A node in the MCTS search tree (shared with standard MCTS)."""

    __slots__ = [
        "board", "parent", "move", "children",
        "visit_count", "total_value", "prior",
        "is_expanded", "is_terminal", "terminal_value",
    ]

    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False
        self.is_terminal = board.is_game_over(claim_draw=True)
        self.terminal_value = None
        if self.is_terminal:
            self._compute_terminal_value()

    def _compute_terminal_value(self):
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            self.terminal_value = 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            self.terminal_value = -1.0 if self.board.turn == chess.WHITE else 1.0
        else:
            self.terminal_value = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class GumbelMCTS:
    """
    Gumbel MuZero-style MCTS with Sequential Halving.

    Instead of PUCT + Dirichlet noise, uses:
    - Gumbel noise for exploration (principled, not heuristic)
    - Sequential Halving to allocate simulations efficiently
    - Completed Q-values with sigma-transform for policy improvement
    """

    def __init__(self, model, device, num_simulations=16, c_visit=50.0, max_num_considered=16):
        """
        Args:
            model: ChessGATv2 neural network
            device: torch device
            num_simulations: total simulation budget (works well with 8-64)
            c_visit: scaling constant for mixing Q-values with priors
            max_num_considered: max actions to consider (top-k by prior + gumbel)
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_visit = c_visit
        self.max_num_considered = max_num_considered

    @torch.no_grad()
    def _evaluate(self, board):
        """Run neural network on a position. Returns (value, policy_dict, log_prior_dict)."""
        fen = board.fen()
        graph = fen_to_graph(fen)
        batch = Batch.from_data_list([graph]).to(self.device)
        value_logits, policy_logits, *_ = self.model(batch)

        # Value: P(win) - P(loss)
        wdl = F.softmax(value_logits, dim=1)[0]
        value = (wdl[0] - wdl[2]).item()

        # Policy: log-softmax over edge logits
        legal_moves = list(board.legal_moves)
        if policy_logits is not None and len(legal_moves) > 0:
            log_probs = F.log_softmax(policy_logits, dim=0).cpu().numpy()
            probs = F.softmax(policy_logits, dim=0).cpu().numpy()
            policy_dict = {move: float(probs[i]) for i, move in enumerate(legal_moves)}
            log_prior_dict = {move: float(log_probs[i]) for i, move in enumerate(legal_moves)}
        else:
            uniform = 1.0 / max(len(legal_moves), 1)
            log_uniform = math.log(uniform) if len(legal_moves) > 0 else 0.0
            policy_dict = {m: uniform for m in legal_moves}
            log_prior_dict = {m: log_uniform for m in legal_moves}

        return value, policy_dict, log_prior_dict

    def _expand(self, node, policy_dict=None, value=None):
        """Expand a leaf node. If policy_dict given, use it; otherwise evaluate."""
        if node.is_terminal:
            return node.terminal_value

        if policy_dict is None:
            value, policy_dict, _ = self._evaluate(node.board)

        for move, prior in policy_dict.items():
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, move=move, prior=prior)

        node.is_expanded = True
        return value

    def _simulate_action(self, root, move):
        """Run one simulation: expand the child if needed, then backprop."""
        child = root.children[move]

        if child.is_terminal:
            value = child.terminal_value
        elif not child.is_expanded:
            value = self._expand(child)
        else:
            # If already expanded, do a full PUCT descent from this child
            leaf = self._select_leaf(child)
            value = self._expand(leaf)
            self._backpropagate(leaf, value)
            return  # backprop already done up to child

        self._backpropagate(child, value)

    def _select_leaf(self, node):
        """Walk down from node picking best UCB child until leaf."""
        while node.is_expanded and not node.is_terminal:
            best_child = None
            best_score = -float("inf")
            total_visits = sum(c.visit_count for c in node.children.values())
            for child in node.children.values():
                exploration = 1.5 * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
                score = child.q_value + exploration
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    def _backpropagate(self, node, value):
        """Walk up the tree, flipping value sign at each level."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    def _sigma_transform(self, q_values, root_value):
        """
        Monotonic normalization of Q-values to [0, 1].
        Unvisited actions get the root value estimate.
        """
        if not q_values:
            return {}

        all_values = list(q_values.values())
        min_q = min(all_values)
        max_q = max(all_values)

        if max_q - min_q < 1e-8:
            return {move: 0.5 for move in q_values}

        return {move: (q - min_q) / (max_q - min_q) for move, q in q_values.items()}

    def _completed_q(self, root, moves, root_value):
        """
        Get completed Q-values: real Q for visited children, root_value for unvisited.
        Note: Q-values are from parent's perspective (negated child Q).
        """
        q_values = {}
        for move in moves:
            child = root.children.get(move)
            if child and child.visit_count > 0:
                # Child's Q is from child's perspective; negate for parent
                q_values[move] = -child.q_value
            else:
                q_values[move] = root_value
        return q_values

    def search(self, board, temperature=1.0):
        """
        Run Gumbel MCTS from the given position.

        Args:
            board: chess.Board
            temperature: controls final policy sharpness (for interface compat)

        Returns:
            move_probs: dict mapping chess.Move -> float (improved policy)
            root: MCTSNode
        """
        root = MCTSNode(board.copy())

        # Evaluate root
        root_value, policy_dict, log_prior_dict = self._evaluate(root.board)
        self._expand(root, policy_dict, root_value)
        root.visit_count = 1
        root.total_value = root_value

        moves = list(root.children.keys())
        n_actions = len(moves)

        if n_actions == 0:
            return {}, root

        if n_actions == 1:
            return {moves[0]: 1.0}, root

        # Sample Gumbel noise for each action
        gumbel_noise = np.random.gumbel(0, 1, size=n_actions)
        log_priors = np.array([log_prior_dict[m] for m in moves])

        # Scores = gumbel + log_prior (for selecting top-k)
        scores = gumbel_noise + log_priors

        # Limit to top-k actions by score (reduces branching)
        k = min(self.max_num_considered, n_actions)
        if k < n_actions:
            top_indices = np.argpartition(scores, -k)[-k:]
        else:
            top_indices = np.arange(n_actions)

        considered_moves = [moves[i] for i in top_indices]
        considered_gumbel = gumbel_noise[top_indices]
        considered_log_priors = log_priors[top_indices]
        m = len(considered_moves)

        # Sequential Halving
        n_sims = self.num_simulations
        if m <= 1:
            num_phases = 1
        else:
            num_phases = max(1, math.ceil(math.log2(m)))

        remaining = list(range(m))  # indices into considered_moves

        for phase in range(num_phases):
            if len(remaining) <= 1:
                break

            # Allocate sims equally among remaining actions
            sims_this_phase = max(1, n_sims // (num_phases * len(remaining)))

            for idx in remaining:
                move = considered_moves[idx]
                for _ in range(sims_this_phase):
                    self._simulate_action(root, move)

            # Compute completed Q-values for remaining actions
            remaining_moves = [considered_moves[i] for i in remaining]
            q_dict = self._completed_q(root, remaining_moves, root_value)
            sigma_q = self._sigma_transform(q_dict, root_value)

            # Score = gumbel + log_prior + sigma(Q) * c_visit
            phase_scores = []
            for idx in remaining:
                move = considered_moves[idx]
                sq = sigma_q.get(move, 0.5)
                score = considered_gumbel[idx] + considered_log_priors[idx] + sq * self.c_visit
                phase_scores.append((idx, score))

            # Keep top half
            phase_scores.sort(key=lambda x: x[1], reverse=True)
            half = max(1, len(phase_scores) // 2)
            remaining = [idx for idx, _ in phase_scores[:half]]

        # Final: use any remaining simulation budget on the survivor(s)
        for idx in remaining:
            move = considered_moves[idx]
            child = root.children[move]
            leftover = max(0, n_sims - sum(c.visit_count for c in root.children.values()))
            for _ in range(leftover):
                self._simulate_action(root, move)

        # Compute improved policy from all considered actions
        all_q = self._completed_q(root, [considered_moves[i] for i in range(m)], root_value)
        all_sigma = self._sigma_transform(all_q, root_value)

        # Improved logits = gumbel + log_prior + sigma(Q) * c_visit
        improved_logits = np.zeros(m)
        for i in range(m):
            move = considered_moves[i]
            improved_logits[i] = (
                considered_gumbel[i]
                + considered_log_priors[i]
                + all_sigma.get(move, 0.5) * self.c_visit
            )

        # Apply temperature
        if temperature < 0.01:
            best_idx = np.argmax(improved_logits)
            move_probs = {considered_moves[i]: (1.0 if i == best_idx else 0.0) for i in range(m)}
        else:
            # Softmax with temperature
            logits = improved_logits / temperature
            logits -= logits.max()  # numerical stability
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()
            move_probs = {considered_moves[i]: float(probs[i]) for i in range(m)}

        # Add zero probs for unconsidered moves (needed for policy targets)
        for i in range(n_actions):
            if moves[i] not in move_probs:
                move_probs[moves[i]] = 0.0

        return move_probs, root
