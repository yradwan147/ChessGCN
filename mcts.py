"""
Monte Carlo Tree Search with neural network guidance (PUCT).
"""

import math

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from data import fen_to_graph


class MCTSNode:
    """A node in the MCTS search tree."""

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
        """Game outcome from the perspective of the side to move."""
        result = self.board.result(claim_draw=True)
        if result == "1-0":
            # White won. Side to move: if white, +1; if black, -1.
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

    def ucb_score(self, c_puct):
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTS:
    """Monte Carlo Tree Search with neural network value + policy guidance."""

    def __init__(self, model, device, c_puct=1.5, num_simulations=128,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def _evaluate(self, board):
        """
        Run neural network on a position.

        Returns:
            value: float in [-1, 1] from perspective of side to move
            policy_dict: dict mapping chess.Move → prior probability
        """
        fen = board.fen()
        graph = fen_to_graph(fen)
        batch = Batch.from_data_list([graph]).to(self.device)
        value_logits, policy_logits, *_ = self.model(batch)

        # Value: P(win) - P(loss)
        wdl = F.softmax(value_logits, dim=1)[0]
        value = (wdl[0] - wdl[2]).item()

        # Policy: softmax over edge logits
        legal_moves = list(board.legal_moves)
        if policy_logits is not None and len(legal_moves) > 0:
            probs = F.softmax(policy_logits, dim=0).cpu().numpy()
            policy_dict = {move: float(probs[i]) for i, move in enumerate(legal_moves)}
        else:
            uniform = 1.0 / max(len(legal_moves), 1)
            policy_dict = {m: uniform for m in legal_moves}

        return value, policy_dict

    def _select(self, node):
        """Walk down the tree picking the child with highest UCB until a leaf."""
        while node.is_expanded and not node.is_terminal:
            node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
        return node

    def _expand(self, node):
        """Expand a leaf node using the neural network. Returns value for backprop."""
        if node.is_terminal:
            return node.terminal_value

        value, policy_dict = self._evaluate(node.board)

        for move, prior in policy_dict.items():
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, move=move, prior=prior)

        node.is_expanded = True
        return value

    def _backpropagate(self, node, value):
        """Walk up the tree, flipping the value sign at each level."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root priors for exploration."""
        if not node.children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]

    def search(self, board, temperature=1.0):
        """
        Run MCTS from the given position.

        Args:
            board: chess.Board
            temperature: 1.0 = proportional to visits, 0 = greedy

        Returns:
            move_probs: dict mapping chess.Move → float (MCTS policy)
            root: MCTSNode (for value inspection)
        """
        root = MCTSNode(board.copy())
        self._expand(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            leaf = self._select(root)
            value = self._expand(leaf)
            self._backpropagate(leaf, value)

        # Extract visit-count distribution
        move_visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(move_visits.values())

        if total_visits == 0:
            n = max(len(move_visits), 1)
            return {m: 1.0 / n for m in move_visits}, root

        if temperature < 0.01:
            # Greedy
            best_move = max(move_visits, key=move_visits.get)
            move_probs = {m: 0.0 for m in move_visits}
            move_probs[best_move] = 1.0
        else:
            visits = np.array([move_visits[m] for m in move_visits], dtype=np.float64)
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            move_probs = {move: float(probs[i]) for i, move in enumerate(move_visits)}

        return move_probs, root
