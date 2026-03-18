"""
Self-play training pipeline for Chess GCN.

Usage:
    # Bootstrap: supervised policy pretraining then self-play
    python selfplay.py --pretrain-policy --checkpoint best_model.pt

    # Self-play from a dual-head checkpoint
    python selfplay.py --checkpoint pretrained_dual_head.pt --iterations 100
"""

import argparse
import copy
import json
import logging
import random
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import fen_to_graph, cp_to_wdl, load_and_sample
from model import ChessGATv2, load_v1_checkpoint
from mcts import MCTS
from gumbel_mcts import GumbelMCTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SelfPlay")


# ── Opening Book ────────────────────────────────────────────────────────────

def load_opening_book(path):
    """Load opening book from a text file.

    Format: Name | uci_move1 uci_move2 ...
    Returns list of (name, result_fen) tuples where result_fen is the
    position after all book moves have been played.
    """
    openings = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            uci_strs = parts[1].strip().split()
            board = chess.Board()
            valid = True
            for uci in uci_strs:
                try:
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        valid = False
                        break
                    board.push(move)
                except ValueError:
                    valid = False
                    break
            if valid and uci_strs:
                openings.append((name, board.fen()))
    return openings


# ── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer with optional sliding window and weighted replay.

    Args:
        max_size: Maximum buffer capacity.
        window: If set, only sample from the last N iterations of data.
        replay_decay: If set, weight samples by decay^(current_iter - sample_iter).
                      1.0 = uniform, 0.9 = recent data 10x more likely than 10-iter-old data.
    """

    def __init__(self, max_size=300_000, window=None, replay_decay=None):
        self.buffer = deque(maxlen=max_size)
        self.window = window
        self.replay_decay = replay_decay

    def add(self, graph, policy_target, value_target, iteration=0,
            moves_left_target=None):
        entry = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            policy_target=policy_target,
            value_target=torch.tensor([value_target], dtype=torch.float),
            iter_num=torch.tensor([iteration], dtype=torch.long),
        )
        if moves_left_target is not None:
            entry.moves_left_target = torch.tensor([moves_left_target], dtype=torch.float)
        self.buffer.append(entry)

    def sample(self, batch_size, current_iter=0):
        candidates = list(self.buffer)

        # Sliding window: only keep recent iterations
        if self.window is not None and current_iter > 0:
            min_iter = max(0, current_iter - self.window)
            candidates = [e for e in candidates if e.iter_num.item() >= min_iter]
            if not candidates:
                candidates = list(self.buffer)  # fallback if window too aggressive

        n = min(batch_size, len(candidates))

        # Weighted replay: recent data sampled more often
        if self.replay_decay is not None and self.replay_decay < 1.0 and current_iter > 0:
            weights = []
            for e in candidates:
                age = current_iter - e.iter_num.item()
                weights.append(self.replay_decay ** age)
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = np.random.choice(len(candidates), size=n, replace=False, p=probs)
        else:
            indices = random.sample(range(len(candidates)), n)

        return [candidates[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ── Per-graph cross-entropy (policy loss) ────────────────────────────────────

def _scatter_log_softmax(logits, batch_idx):
    """Numerically stable log-softmax grouped by batch_idx."""
    # Max per graph for stability
    num_graphs = batch_idx.max().item() + 1
    max_vals = torch.full((num_graphs,), -1e9, device=logits.device)
    max_vals.scatter_reduce_(0, batch_idx, logits, reduce="amax")
    shifted = logits - max_vals[batch_idx]
    exp_vals = shifted.exp()
    sum_exp = torch.zeros(num_graphs, device=logits.device)
    sum_exp.scatter_add_(0, batch_idx, exp_vals)
    return shifted - sum_exp[batch_idx].log()


def per_graph_cross_entropy(logits, targets, batch_idx):
    """Cross-entropy where softmax is computed per-graph, not globally."""
    log_probs = _scatter_log_softmax(logits, batch_idx)
    num_graphs = batch_idx.max().item() + 1
    return -(targets * log_probs).sum() / num_graphs


# ── Self-play game generation ────────────────────────────────────────────────

def play_one_game(mcts_engine, max_moves=512, temp_threshold=30,
                  resign_threshold=-0.95, resign_enabled=True,
                  start_fen=None, fast_engine=None):
    """
    Play a single self-play game.

    Args:
        fast_engine: If provided, enables playout cap randomization.
                     75% of moves use fast_engine (no policy target),
                     25% use mcts_engine (full search, policy target recorded).

    Returns:
        game_data: list of (fen, mcts_policy_dict, side_to_move, is_full) tuples
        game_record: list of dicts with per-move info for game logging
        result: 1.0 (White wins), 0.0 (draw), -1.0 (Black wins)
    """
    board = chess.Board(start_fen) if start_fen else chess.Board()
    game_data = []
    game_record = []

    for move_num in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        # Playout cap: 75% fast search, 25% full search
        is_full_search = True
        if fast_engine and random.random() < 0.75:
            is_full_search = False
            move_probs, root = fast_engine.search(board, temperature=0.01)
        else:
            # Smooth temperature annealing: 1.0 → 0.3 linearly, then greedy
            if move_num < temp_threshold:
                temperature = max(0.3, 1.0 - 0.7 * move_num / temp_threshold)
            else:
                temperature = 0.01
            move_probs, root = mcts_engine.search(board, temperature=temperature)

        game_data.append((board.fen(), move_probs, board.turn, is_full_search))

        # Build top-5 move info from MCTS visit counts
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top5 = []
        for move, prob in sorted_moves:
            child = root.children.get(move)
            top5.append({
                "uci": move.uci(),
                "san": board.san(move),
                "visits": child.visit_count if child else 0,
                "prob": round(prob, 4),
                "q": round(child.q_value, 4) if child and child.visit_count > 0 else 0.0,
            })

        # Resignation check
        if resign_enabled and root.q_value < resign_threshold:
            game_record.append({
                "fen": board.fen(),
                "side": "white" if board.turn == chess.WHITE else "black",
                "move": "resign",
                "top5": top5,
                "root_q": round(root.q_value, 4),
            })
            result = -1.0 if board.turn == chess.WHITE else 1.0
            return game_data, game_record, result

        # Sample move from MCTS policy
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])
        probs = probs / probs.sum()
        chosen = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen]

        game_record.append({
            "fen": board.fen(),
            "side": "white" if board.turn == chess.WHITE else "black",
            "move": chosen_move.uci(),
            "move_san": board.san(chosen_move),
            "top5": top5,
            "root_q": round(root.q_value, 4),
        })

        board.push(chosen_move)

    # Game ended
    if board.is_game_over(claim_draw=True):
        r = board.result(claim_draw=True)
        result = 1.0 if r == "1-0" else (-1.0 if r == "0-1" else 0.0)
    else:
        result = 0.0  # max moves reached

    return game_data, game_record, result


def add_game_to_buffer(game_data, result, replay_buffer, iteration=0,
                       color_flip=False, moves_left=False, max_moves=200):
    """Convert a self-play game into training samples and add to buffer.

    If color_flip=True, also adds a color-flipped copy of each position
    (value only, no policy target) for data augmentation.
    If moves_left=True, stores normalized moves-to-end as auxiliary target.
    """
    total_moves = len(game_data)
    for move_idx, entry in enumerate(game_data):
        fen, move_probs, side_to_move = entry[0], entry[1], entry[2]
        is_full = entry[3] if len(entry) > 3 else True

        graph = fen_to_graph(fen)
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        # Only record policy target for full-search positions
        if is_full:
            policy_target = torch.zeros(len(legal_moves), dtype=torch.float)
            for i, move in enumerate(legal_moves):
                policy_target[i] = move_probs.get(move, 0.0)
            if policy_target.sum() > 0:
                policy_target = policy_target / policy_target.sum()
        else:
            policy_target = torch.zeros(len(legal_moves), dtype=torch.float)

        # Value from perspective of the side to move
        value = result if side_to_move == chess.WHITE else -result

        # Moves-left target (normalized to [0, 1])
        ml_target = None
        if moves_left:
            ml_target = (total_moves - move_idx) / max_moves

        replay_buffer.add(graph, policy_target, value, iteration=iteration,
                          moves_left_target=ml_target)

        # Color flip augmentation: add mirrored position (value only)
        if color_flip and random.random() < 0.5:
            mirrored_board = board.mirror()
            mirrored_graph = fen_to_graph(mirrored_board.fen())
            mirrored_policy = torch.zeros(
                len(list(mirrored_board.legal_moves)), dtype=torch.float)
            replay_buffer.add(mirrored_graph, mirrored_policy, -value,
                              iteration=iteration, moves_left_target=ml_target)


# ── Training on replay buffer ────────────────────────────────────────────────

def train_on_buffer(model, replay_buffer, optimizer, device,
                    batch_size=256, num_batches=200, current_iter=0):
    """Train with combined value + policy loss on replay buffer samples."""
    model.train()
    total_vloss, total_ploss, n = 0.0, 0.0, 0

    for _ in range(num_batches):
        if len(replay_buffer) < batch_size:
            break
        samples = replay_buffer.sample(batch_size, current_iter=current_iter)
        batch = Batch.from_data_list(samples).to(device)
        optimizer.zero_grad()

        value_logits, policy_logits, moves_left_pred = model(batch)

        # --- Value loss ---
        z = batch.value_target.view(-1)
        n_val = value_logits.shape[1]
        if n_val == 3:
            # Standard WDL loss
            target_wdl = torch.zeros(z.size(0), 3, device=z.device)
            target_wdl[:, 0] = (z == 1).float()    # win
            target_wdl[:, 1] = (z == 0).float()    # draw
            target_wdl[:, 2] = (z == -1).float()   # loss
            log_probs = F.log_softmax(value_logits, dim=1)
            value_loss = F.kl_div(log_probs, target_wdl, reduction="batchmean")
        else:
            # Score distribution (N bins): HL-Gauss soft targets
            bin_centers = torch.linspace(0, 1, n_val, device=z.device)
            bin_width = bin_centers[1] - bin_centers[0]
            sigma = 0.75 * bin_width
            # Convert z ∈ {-1, 0, 1} to win probability ∈ [0, 1]
            win_pct = (z + 1.0) / 2.0  # -1→0, 0→0.5, 1→1
            # HL-Gauss soft target distribution
            target_dist = torch.exp(-0.5 * ((bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)) / sigma) ** 2)
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
            log_probs = F.log_softmax(value_logits, dim=1)
            value_loss = F.kl_div(log_probs, target_dist, reduction="batchmean")

        # --- Policy loss (only on legal-move edges, not self-edges) ---
        if policy_logits is not None and policy_logits.numel() > 0:
            # edge_batch must match policy_logits (self-edges filtered in model)
            move_mask = batch.edge_index[0] != batch.edge_index[1]
            edge_batch = batch.batch[batch.edge_index[0][move_mask]]
            policy_loss = per_graph_cross_entropy(
                policy_logits, batch.policy_target, edge_batch,
            )
        else:
            policy_loss = torch.tensor(0.0, device=device)

        loss = policy_loss + value_loss

        # --- Moves-left auxiliary loss ---
        if moves_left_pred is not None and hasattr(batch, 'moves_left_target'):
            ml_target = batch.moves_left_target.view(-1)
            ml_loss = F.mse_loss(moves_left_pred.squeeze(-1), ml_target)
            loss = loss + 0.002 * ml_loss  # Czech et al. weight

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_vloss += value_loss.item()
        total_ploss += policy_loss.item()
        n += 1

    return {
        "value_loss": total_vloss / max(n, 1),
        "policy_loss": total_ploss / max(n, 1),
    }


# ── Evaluation: challenger vs best model ─────────────────────────────────────

def evaluate_models(challenger, best_model, device,
                    num_games=40, num_simulations=64, max_moves=200,
                    use_gumbel=False):
    """Play challenger vs best. Returns challenger win rate."""
    challenger.eval()
    best_model.eval()
    EngineClass = GumbelMCTS if use_gumbel else MCTS
    c_mcts = EngineClass(challenger, device, num_simulations=num_simulations)
    b_mcts = EngineClass(best_model, device, num_simulations=num_simulations)

    challenger_points = 0.0

    for game_idx in range(num_games):
        challenger_is_white = (game_idx % 2 == 0)
        white_mcts = c_mcts if challenger_is_white else b_mcts
        black_mcts = b_mcts if challenger_is_white else c_mcts

        board = chess.Board()
        for _ in range(max_moves):
            if board.is_game_over(claim_draw=True):
                break
            current = white_mcts if board.turn == chess.WHITE else black_mcts
            move_probs, _ = current.search(board, temperature=0.01)
            best_move = max(move_probs, key=move_probs.get)
            board.push(best_move)

        r = board.result(claim_draw=True)
        if r == "1-0":
            challenger_points += 1.0 if challenger_is_white else 0.0
        elif r == "0-1":
            challenger_points += 0.0 if challenger_is_white else 1.0
        else:
            challenger_points += 0.5

    return challenger_points / num_games


# ── Supervised policy pretraining (bootstrap) ────────────────────────────────

def pretrain_policy_head(args, device):
    """Pre-train the policy head on Stockfish best moves before self-play."""
    log.info("=== Supervised Policy Pretraining ===")

    node_dim = 21 if args.no_check_feature else 22
    model = ChessGATv2(
        node_dim=node_dim,
        hidden=args.hidden, heads=args.heads, num_blocks=args.blocks,
        policy_head=True,
        attn_pool=args.attn_pool,
        moves_left_head=args.moves_left_head,
        score_dist_bins=args.score_dist_bins,
    ).to(device)

    if Path(args.checkpoint).exists():
        load_v1_checkpoint(model, args.checkpoint, device)
        log.info(f"Loaded value-head checkpoint: {args.checkpoint}")

    # Build dataset with policy targets (stratified sampling if subsampling)
    label = f"{args.pretrain_samples}" if args.pretrain_samples else "all"
    log.info(f"Loading {label} samples from {args.pretrain_data}...")
    df = load_and_sample(args.pretrain_data, num_samples=args.pretrain_samples)

    graphs = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building policy dataset"):
        fen = row["Position"]
        wdl = cp_to_wdl(row["cp"])
        best_move_uci = str(row["Best Move"]).strip()

        graph = fen_to_graph(fen, wdl)
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        target_idx = -1
        for i, move in enumerate(legal_moves):
            if move.uci() == best_move_uci:
                target_idx = i
                break

        if target_idx == -1:
            skipped += 1
            continue

        policy_target = torch.zeros(len(legal_moves), dtype=torch.float)
        policy_target[target_idx] = 1.0
        graph.policy_target = policy_target
        graphs.append(graph)

    log.info(f"Built {len(graphs)} graphs ({skipped} skipped)")

    train_g, val_g = train_test_split(graphs, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_g, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_g, batch_size=args.batch_size)

    # Differential LR: slow for backbone, fast for policy head
    backbone_params = [p for n, p in model.named_parameters() if "policy_mlp" not in n]
    policy_params = [p for n, p in model.named_parameters() if "policy_mlp" in n]
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": policy_params, "lr": args.lr * 10},
    ], weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        t_vloss, t_ploss, batches = 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            value_logits, policy_logits, *_ = model(batch)

            # Value loss
            target_wdl = batch.y.view(-1, 3)
            log_probs = F.log_softmax(value_logits, dim=1)
            value_loss = F.kl_div(log_probs, target_wdl, reduction="batchmean")

            # Policy loss (filter self-edges)
            if policy_logits is not None and policy_logits.numel() > 0:
                move_mask = batch.edge_index[0] != batch.edge_index[1]
                edge_batch = batch.batch[batch.edge_index[0][move_mask]]
                policy_loss = per_graph_cross_entropy(
                    policy_logits, batch.policy_target, edge_batch,
                )
            else:
                policy_loss = torch.tensor(0.0, device=device)

            loss = value_loss + policy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_vloss += value_loss.item()
            t_ploss += policy_loss.item()
            batches += 1

        # Validation
        model.eval()
        v_vloss, v_ploss, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                value_logits, policy_logits, *_ = model(batch)

                target_wdl = batch.y.view(-1, 3)
                log_probs = F.log_softmax(value_logits, dim=1)
                v_vloss += F.kl_div(log_probs, target_wdl, reduction="batchmean").item()

                if policy_logits is not None and policy_logits.numel() > 0:
                    move_mask = batch.edge_index[0] != batch.edge_index[1]
                    edge_batch = batch.batch[batch.edge_index[0][move_mask]]
                    v_ploss += per_graph_cross_entropy(
                        policy_logits, batch.policy_target, edge_batch,
                    ).item()
                v_batches += 1

        avg_v = v_vloss / max(v_batches, 1)
        avg_p = v_ploss / max(v_batches, 1)
        marker = ""
        if (avg_v + avg_p) < best_val_loss:
            best_val_loss = avg_v + avg_p
            marker = " *"

        log.info(
            f"Pretrain {epoch}/{args.pretrain_epochs} | "
            f"Train vl={t_vloss/batches:.4f} pl={t_ploss/batches:.4f} | "
            f"Val vl={avg_v:.4f} pl={avg_p:.4f}{marker}"
        )

    out_path = "pretrained_dual_head.pt"
    torch.save(model.state_dict(), out_path)
    log.info(f"Saved pretrained dual-head model to {out_path}")
    return model


# ── Main self-play loop ──────────────────────────────────────────────────────

def selfplay_main(args, model=None):
    device = get_device()
    log.info(f"Device: {device}")

    # Override data.py defaults based on CLI flags
    import data as data_module
    if args.no_self_edges:
        # Monkey-patch fen_to_graph to disable self-edges
        _original_ftg = data_module.fen_to_graph
        data_module.fen_to_graph = lambda fen, wdl=None, **kw: _original_ftg(
            fen, wdl, self_edges=False, check_feature=not args.no_check_feature)
        globals()['fen_to_graph'] = data_module.fen_to_graph
    elif args.no_check_feature:
        _original_ftg = data_module.fen_to_graph
        data_module.fen_to_graph = lambda fen, wdl=None, **kw: _original_ftg(
            fen, wdl, check_feature=False)
        globals()['fen_to_graph'] = data_module.fen_to_graph
    if args.wdl_k != 200.0:
        data_module.WDL_K = args.wdl_k
        log.info(f"WDL_K set to {args.wdl_k}")

    if model is None:
        node_dim = 21 if args.no_check_feature else 22
        model = ChessGATv2(
            node_dim=node_dim,
            hidden=args.hidden, heads=args.heads, num_blocks=args.blocks,
            policy_head=True,
            attn_pool=args.attn_pool,
            moves_left_head=args.moves_left_head,
            score_dist_bins=args.score_dist_bins,
        ).to(device)

        ckpt = args.checkpoint
        if Path("pretrained_dual_head.pt").exists() and args.pretrain_policy:
            ckpt = "pretrained_dual_head.pt"

        if Path(ckpt).exists():
            state = torch.load(ckpt, weights_only=True, map_location=device)
            model.load_state_dict(state, strict=False)
            log.info(f"Loaded checkpoint: {ckpt}")

    best_model = copy.deepcopy(model)
    best_model.eval()
    replay_buffer = ReplayBuffer(
        max_size=args.buffer_size,
        window=args.buffer_window,
        replay_decay=args.replay_decay,
    )

    # Optimizer with differential LR
    backbone_params = [p for n, p in model.named_parameters()
                       if "policy_mlp" not in n and "value_head" not in n]
    policy_params = [p for n, p in model.named_parameters()
                     if "policy_mlp" in n]

    if args.freeze_value:
        for param in model.value_head.parameters():
            param.requires_grad = False
        optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": policy_params, "lr": args.lr},
        ], weight_decay=1e-4)
        log.info("Value head frozen (supervised weights preserved)")
    else:
        value_params = [p for n, p in model.named_parameters()
                        if "value_head" in n]
        optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": policy_params, "lr": args.lr},
            {"params": value_params, "lr": args.lr * 0.01},
        ], weight_decay=1e-4)
        log.info("Value head unfrozen with tiny LR (0.01x)")

    # Build FEN pool for diverse starting positions
    fen_pool = []
    if Path(args.pretrain_data).exists():
        fen_df = load_and_sample(args.pretrain_data, num_samples=None)
        balanced = fen_df[fen_df["cp"].abs() < 500]
        fen_pool = balanced["Position"].tolist()
        if len(fen_pool) > 10_000:
            fen_pool = random.sample(fen_pool, 10_000)
        log.info(f"Built FEN pool: {len(fen_pool)} balanced positions for diverse starts")

    # Load opening book for opening suite starts
    opening_book = []
    if args.openings_file and Path(args.openings_file).exists():
        opening_book = load_opening_book(args.openings_file)
        log.info(f"Loaded opening book: {len(opening_book)} lines from {args.openings_file}")

    # Game log file (JSON lines format)
    games_file = Path(args.games_file)
    log.info(f"Saving games to {games_file}")

    # EMA model for stable self-play game generation
    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        log.info("EMA model enabled (decay=0.995)")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {num_params:,} trainable parameters")
    log.info(f"Self-play: {args.iterations} iterations, {args.games_per_iter} games/iter, "
             f"{args.simulations} MCTS sims/move")

    consecutive_bad_evals = 0

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"Iteration {iteration}/{args.iterations}")
        log.info(f"{'='*60}")

        # ── Phase 1: Self-play ──
        # Use EMA model for game generation if available (more stable)
        play_model = ema_model if ema_model else model
        play_model.eval()
        if args.use_gumbel:
            mcts_engine = GumbelMCTS(play_model, device, num_simulations=args.simulations)
        else:
            mcts_engine = MCTS(play_model, device, num_simulations=args.simulations)

        # Fast engine for playout cap randomization (fewer sims, no noise)
        fast_engine = None
        if args.playout_cap:
            if args.use_gumbel:
                fast_engine = GumbelMCTS(play_model, device, num_simulations=args.fast_sims)
            else:
                fast_engine = MCTS(play_model, device, num_simulations=args.fast_sims)

        outcomes = {"W": 0, "D": 0, "B": 0}
        positions_added = 0

        for g in range(args.games_per_iter):
            resign_on = random.random() > 0.2  # 80% with resign, 20% without

            # Game start selection: opening suite > FEN pool > standard
            start_fen = None
            game_type = "standard"
            roll = random.random()
            if opening_book and roll < args.opening_ratio:
                name, fen = random.choice(opening_book)
                start_fen = fen
                game_type = f"book:{name}"
            elif fen_pool and roll < args.opening_ratio + args.fen_pool_ratio:
                start_fen = random.choice(fen_pool)
                game_type = "fen_pool"

            game_data, game_record, result = play_one_game(
                mcts_engine, max_moves=args.max_moves, resign_enabled=resign_on,
                start_fen=start_fen, fast_engine=fast_engine,
            )
            add_game_to_buffer(game_data, result, replay_buffer,
                               iteration=iteration, color_flip=args.color_flip,
                               moves_left=args.moves_left_head,
                               max_moves=args.max_moves)
            positions_added += len(game_data)

            # Save game record to file
            result_str = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}.get(result, "?")
            game_entry = {
                "iteration": iteration,
                "game": g + 1,
                "result": result_str,
                "num_moves": len(game_record),
                "game_type": game_type,
                "moves": game_record,
            }
            with open(games_file, "a") as f:
                f.write(json.dumps(game_entry) + "\n")

            outcome_key = {1.0: "W", -1.0: "B", 0.0: "D"}.get(result, "?")
            outcomes[outcome_key] = outcomes.get(outcome_key, 0) + 1

            if (g + 1) % max(1, args.games_per_iter // 5) == 0:
                log.info(f"  Games {g+1}/{args.games_per_iter} | "
                         f"W:{outcomes['W']} D:{outcomes['D']} B:{outcomes['B']} | "
                         f"Buffer: {len(replay_buffer):,}")

        log.info(f"  Self-play done: {positions_added} positions from "
                 f"{args.games_per_iter} games ({outcomes})")

        # ── Phase 2: Training (scale steps to avoid overfitting on small buffer) ──
        actual_steps = min(args.train_steps, len(replay_buffer) // args.batch_size)
        model.train()
        losses = train_on_buffer(
            model, replay_buffer, optimizer, device,
            batch_size=args.batch_size, num_batches=actual_steps,
            current_iter=iteration,
        )
        log.info(f"  Training ({actual_steps} steps): value_loss={losses['value_loss']:.4f}, "
                 f"policy_loss={losses['policy_loss']:.4f}")

        # Update EMA model
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(0.995).add_(p.data, alpha=0.005)

        # Save checkpoint every iteration (crash resilience)
        torch.save(model.state_dict(), "selfplay_latest.pt")
        if iteration % args.save_interval == 0:
            torch.save(model.state_dict(), f"selfplay_iter{iteration}.pt")

        # ── Phase 3: Evaluation (every K iterations) ──
        if iteration % args.eval_interval == 0:
            log.info(f"  Evaluating challenger vs best ({args.eval_games} games)...")
            eval_model = ema_model if ema_model else model
            eval_model.eval()
            win_rate = evaluate_models(
                eval_model, best_model, device,
                num_games=args.eval_games,
                num_simulations=args.eval_sims,
                max_moves=args.max_moves,
                use_gumbel=args.use_gumbel,
            )
            log.info(f"  Challenger win rate: {win_rate:.1%}")
            if win_rate >= args.gate_threshold:
                log.info(f"  New best model!")
                best_model = copy.deepcopy(eval_model)
                best_model.eval()
                torch.save(eval_model.state_dict(), f"selfplay_best_iter{iteration}.pt")
                consecutive_bad_evals = 0
            elif args.restore_on_collapse and win_rate < 0.35:
                consecutive_bad_evals += 1
                log.info(f"  Low eval ({win_rate:.1%}), bad streak: {consecutive_bad_evals}")
                if consecutive_bad_evals >= 2:
                    log.info(f"  Collapse detected! Restoring best model weights.")
                    model.load_state_dict(best_model.state_dict())
                    consecutive_bad_evals = 0
            else:
                log.info(f"  Keeping previous best model.")
                consecutive_bad_evals = 0

        elapsed = time.time() - iter_start
        log.info(f"  Iteration time: {elapsed:.0f}s")

    # Save final model
    torch.save(model.state_dict(), "selfplay_final.pt")
    log.info("Self-play training complete. Final model saved to selfplay_final.pt")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-play training for Chess GCN")

    # Model (defaults: best architecture from sweep)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")

    # Self-play
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--games-per-iter", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Max half-moves per self-play game")
    parser.add_argument("--buffer-size", type=int, default=300_000)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=200)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--eval-sims", type=int, default=64,
                        help="MCTS simulations for evaluation games")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--gate-threshold", type=float, default=0.52,
                        help="Win rate threshold for replacing best model")
    parser.add_argument("--games-file", type=str, default="selfplay_games.jsonl",
                        help="Path to save game records (JSON lines)")

    # MCTS variant
    parser.add_argument("--use-gumbel", action="store_true",
                        help="Use Gumbel MuZero MCTS instead of standard PUCT")
    parser.add_argument("--freeze-value", action="store_true",
                        help="Freeze value head during self-play (preserve supervised weights)")

    # Opening suite
    parser.add_argument("--openings-file", type=str, default=None,
                        help="Path to opening book file (pipe-separated name|uci_moves)")
    parser.add_argument("--opening-ratio", type=float, default=0.7,
                        help="Fraction of games starting from opening suite positions")
    parser.add_argument("--fen-pool-ratio", type=float, default=0.1,
                        help="Fraction of games from FEN pool (remainder = standard position)")

    # Research-driven improvements
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weight averaging for game generation (decay=0.995)")
    parser.add_argument("--color-flip", action="store_true",
                        help="Color flip augmentation (adds mirrored positions to buffer)")
    parser.add_argument("--playout-cap", action="store_true",
                        help="Playout cap randomization (75%% fast 8 sims / 25%% full)")
    parser.add_argument("--fast-sims", type=int, default=8,
                        help="Simulations for fast search in playout cap")
    parser.add_argument("--no-self-edges", action="store_true",
                        help="Disable self-edges in graph construction")
    parser.add_argument("--no-check-feature", action="store_true",
                        help="Disable is-in-check node feature")
    parser.add_argument("--wdl-k", type=float, default=200.0,
                        help="WDL_K constant for centipawn to WDL conversion")

    # Anti-collapse
    parser.add_argument("--buffer-window", type=int, default=10,
                        help="Only sample from last N iterations of data (None = no window)")
    parser.add_argument("--replay-decay", type=float, default=None,
                        help="Exponential decay for replay weighting (e.g. 0.9). None = uniform.")
    parser.add_argument("--restore-on-collapse", action="store_true",
                        help="Restore best model weights if eval drops below 35%% twice")

    # Architecture (Phase 2)
    parser.add_argument("--attn-pool", action="store_true",
                        help="Use global attention pooling instead of mean pooling")
    parser.add_argument("--moves-left-head", action="store_true",
                        help="Enable moves-left auxiliary head (Czech et al. +33 Elo)")
    parser.add_argument("--score-dist-bins", type=int, default=0,
                        help="Score distribution bins (0 = use WDL, 128 = recommended)")

    # Bootstrap
    parser.add_argument("--pretrain-policy", action="store_true",
                        help="Run supervised policy pretraining first")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-data", type=str, default="dataset_eval.csv")
    parser.add_argument("--pretrain-samples", type=int, default=None,
                        help="Number of samples for policy pretraining (None = full dataset)")

    args = parser.parse_args()
    device = get_device()

    model = None
    if args.pretrain_policy:
        model = pretrain_policy_head(args, device)

    selfplay_main(args, model)


if __name__ == "__main__":
    main()
