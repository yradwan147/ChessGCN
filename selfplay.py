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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SelfPlay")


# ── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer of (graph, policy_target, value_target) training samples."""

    def __init__(self, max_size=300_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, graph, policy_target, value_target):
        entry = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            policy_target=policy_target,
            value_target=torch.tensor([value_target], dtype=torch.float),
        )
        self.buffer.append(entry)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in indices]

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
                  resign_threshold=-0.95, resign_enabled=True):
    """
    Play a single self-play game.

    Returns:
        game_data: list of (fen, mcts_policy_dict, side_to_move) tuples
        game_record: list of dicts with per-move info for game logging
        result: 1.0 (White wins), 0.0 (draw), -1.0 (Black wins)
    """
    board = chess.Board()
    game_data = []
    game_record = []

    for move_num in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        temperature = 1.0 if move_num < temp_threshold else 0.01
        move_probs, root = mcts_engine.search(board, temperature=temperature)

        game_data.append((board.fen(), move_probs, board.turn))

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


def add_game_to_buffer(game_data, result, replay_buffer):
    """Convert a self-play game into training samples and add to buffer."""
    for fen, move_probs, side_to_move in game_data:
        graph = fen_to_graph(fen)
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        policy_target = torch.zeros(len(legal_moves), dtype=torch.float)
        for i, move in enumerate(legal_moves):
            policy_target[i] = move_probs.get(move, 0.0)
        if policy_target.sum() > 0:
            policy_target = policy_target / policy_target.sum()

        # Value from perspective of the side to move
        value = result if side_to_move == chess.WHITE else -result

        replay_buffer.add(graph, policy_target, value)


# ── Training on replay buffer ────────────────────────────────────────────────

def train_on_buffer(model, replay_buffer, optimizer, device,
                    batch_size=256, num_batches=200):
    """Train with combined value + policy loss on replay buffer samples."""
    model.train()
    total_vloss, total_ploss, n = 0.0, 0.0, 0

    for _ in range(num_batches):
        if len(replay_buffer) < batch_size:
            break
        samples = replay_buffer.sample(batch_size)
        batch = Batch.from_data_list(samples).to(device)
        optimizer.zero_grad()

        value_logits, policy_logits = model(batch)

        # --- Value loss ---
        z = batch.value_target.view(-1)
        target_wdl = torch.stack([
            (z + 1) / 2,      # P(win)
            1 - z.abs(),       # P(draw)
            (1 - z) / 2,      # P(loss)
        ], dim=1)
        log_probs = F.log_softmax(value_logits, dim=1)
        value_loss = F.kl_div(log_probs, target_wdl, reduction="batchmean")

        # --- Policy loss ---
        if policy_logits is not None and policy_logits.numel() > 0:
            edge_batch = batch.batch[batch.edge_index[0]]
            policy_loss = per_graph_cross_entropy(
                policy_logits, batch.policy_target, edge_batch,
            )
        else:
            policy_loss = torch.tensor(0.0, device=device)

        loss = value_loss + policy_loss
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
                    num_games=40, num_simulations=64):
    """Play challenger vs best. Returns challenger win rate."""
    challenger.eval()
    best_model.eval()
    c_mcts = MCTS(challenger, device, num_simulations=num_simulations)
    b_mcts = MCTS(best_model, device, num_simulations=num_simulations)

    challenger_points = 0.0

    for game_idx in range(num_games):
        challenger_is_white = (game_idx % 2 == 0)
        white_mcts = c_mcts if challenger_is_white else b_mcts
        black_mcts = b_mcts if challenger_is_white else c_mcts

        board = chess.Board()
        for _ in range(512):
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

    model = ChessGATv2(
        hidden=args.hidden, heads=args.heads, num_blocks=args.blocks,
        policy_head=True,
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

            value_logits, policy_logits = model(batch)

            # Value loss
            target_wdl = batch.y.view(-1, 3)
            log_probs = F.log_softmax(value_logits, dim=1)
            value_loss = F.kl_div(log_probs, target_wdl, reduction="batchmean")

            # Policy loss
            if policy_logits is not None and policy_logits.numel() > 0:
                edge_batch = batch.batch[batch.edge_index[0]]
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
                value_logits, policy_logits = model(batch)

                target_wdl = batch.y.view(-1, 3)
                log_probs = F.log_softmax(value_logits, dim=1)
                v_vloss += F.kl_div(log_probs, target_wdl, reduction="batchmean").item()

                if policy_logits is not None and policy_logits.numel() > 0:
                    edge_batch = batch.batch[batch.edge_index[0]]
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

    if model is None:
        model = ChessGATv2(
            hidden=args.hidden, heads=args.heads, num_blocks=args.blocks,
            policy_head=True,
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
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Game log file (JSON lines format)
    games_file = Path(args.games_file)
    log.info(f"Saving games to {games_file}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {num_params:,} parameters (policy_head=True)")
    log.info(f"Self-play: {args.iterations} iterations, {args.games_per_iter} games/iter, "
             f"{args.simulations} MCTS sims/move")

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"Iteration {iteration}/{args.iterations}")
        log.info(f"{'='*60}")

        # ── Phase 1: Self-play ──
        model.eval()
        mcts_engine = MCTS(model, device, num_simulations=args.simulations)

        outcomes = {"W": 0, "D": 0, "B": 0}
        positions_added = 0

        for g in range(args.games_per_iter):
            resign_on = random.random() > 0.2  # 80% with resign, 20% without
            game_data, game_record, result = play_one_game(
                mcts_engine, resign_enabled=resign_on,
            )
            add_game_to_buffer(game_data, result, replay_buffer)
            positions_added += len(game_data)

            # Save game record to file
            result_str = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}.get(result, "?")
            game_entry = {
                "iteration": iteration,
                "game": g + 1,
                "result": result_str,
                "num_moves": len(game_record),
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

        # ── Phase 2: Training ──
        model.train()
        losses = train_on_buffer(
            model, replay_buffer, optimizer, device,
            batch_size=args.batch_size, num_batches=args.train_steps,
        )
        log.info(f"  Training: value_loss={losses['value_loss']:.4f}, "
                 f"policy_loss={losses['policy_loss']:.4f}")

        # ── Phase 3: Evaluation (every K iterations) ──
        if iteration % args.eval_interval == 0:
            log.info(f"  Evaluating challenger vs best ({args.eval_games} games)...")
            model.eval()
            win_rate = evaluate_models(
                model, best_model, device,
                num_games=args.eval_games,
                num_simulations=max(args.simulations // 2, 16),
            )
            log.info(f"  Challenger win rate: {win_rate:.1%}")
            if win_rate >= 0.55:
                log.info(f"  New best model!")
                best_model = copy.deepcopy(model)
                best_model.eval()
                torch.save(model.state_dict(), f"selfplay_best_iter{iteration}.pt")
            else:
                log.info(f"  Keeping previous best model.")

        # Periodic checkpoint
        if iteration % args.save_interval == 0:
            torch.save(model.state_dict(), f"selfplay_iter{iteration}.pt")

        elapsed = time.time() - iter_start
        log.info(f"  Iteration time: {elapsed:.0f}s")

    # Save final model
    torch.save(model.state_dict(), "selfplay_final.pt")
    log.info("Self-play training complete. Final model saved to selfplay_final.pt")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-play training for Chess GCN")

    # Model
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")

    # Self-play
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=300_000)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=200)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--games-file", type=str, default="selfplay_games.jsonl",
                        help="Path to save game records (JSON lines)")

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
