"""
Data pipeline for Chess GCN: FEN → Graph conversion, WDL labels, caching.
"""

import re
import math
import chess
import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm


# Stockfish WDL calibration constant (centipawns → win probability)
WDL_K = 111.0


def parse_evaluation(eval_str):
    """Convert evaluation string to centipawns. Mate → ±10000 scaled by distance."""
    mate_match = re.match(r"M(-?\d+)", str(eval_str))
    if mate_match:
        mate_moves = int(mate_match.group(1))
        # Closer mates get higher absolute scores
        if mate_moves > 0:
            return 10000 - (mate_moves - 1) * 10  # M1=10000, M2=9990, ...
        else:
            return -10000 + (abs(mate_moves) - 1) * 10
    return float(eval_str)


def cp_to_wdl(cp):
    """Convert centipawn evaluation to (win, draw, loss) probabilities."""
    # For extreme evaluations (mate), use hard labels
    if cp >= 9000:
        return (1.0, 0.0, 0.0)
    if cp <= -9000:
        return (0.0, 0.0, 1.0)

    win = 1.0 / (1.0 + math.exp(-cp / WDL_K))
    loss = 1.0 / (1.0 + math.exp(cp / WDL_K))
    draw = 1.0 - win - loss
    # Clamp draw to avoid negative values from floating point
    draw = max(draw, 0.0)
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def fen_to_graph(fen, wdl):
    """
    Convert a FEN string to a PyG Data object.

    Node features (21d per square):
      - Piece one-hot (12d): WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK
      - Empty flag (1d)
      - File (1d, normalized)
      - Rank (1d, normalized)
      - Side to move (1d, broadcast)
      - Castling rights (4d, broadcast)
      - Half-move clock (1d, normalized, broadcast)

    Edge features (6d per legal move):
      - is_capture, is_promotion, is_castling, is_en_passant
      - dx, dy (file/rank displacement, normalized by 7)
    """
    board = chess.Board(fen)

    # --- Global features (broadcast to all 64 nodes) ---
    side_to_move = 1.0 if board.turn == chess.WHITE else 0.0
    castling = [
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ]
    halfmove = min(board.halfmove_clock, 100) / 100.0

    # Piece one-hot indices: WP=0,WN=1,WB=2,WR=3,WQ=4,WK=5, BP=6,...,BK=11
    PIECE_OFFSET = {chess.WHITE: 0, chess.BLACK: 6}

    # --- Build node features ---
    node_features = []
    for square in chess.SQUARES:  # 0..63 (a1..h8)
        feat = [0.0] * 21

        piece = board.piece_at(square)
        if piece:
            idx = PIECE_OFFSET[piece.color] + (piece.piece_type - 1)  # piece_type: 1-6
            feat[idx] = 1.0
            feat[12] = 0.0  # not empty
        else:
            feat[12] = 1.0  # empty square

        feat[13] = chess.square_file(square) / 7.0  # file
        feat[14] = chess.square_rank(square) / 7.0  # rank
        feat[15] = side_to_move
        feat[16] = castling[0]
        feat[17] = castling[1]
        feat[18] = castling[2]
        feat[19] = castling[3]
        feat[20] = halfmove

        node_features.append(feat)

    # --- Build edges from legal moves ---
    src, dst = [], []
    edge_attrs = []

    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square

        src.append(from_sq)
        dst.append(to_sq)

        dx = (chess.square_file(to_sq) - chess.square_file(from_sq)) / 7.0
        dy = (chess.square_rank(to_sq) - chess.square_rank(from_sq)) / 7.0

        edge_attrs.append([
            float(board.is_capture(move)),
            float(move.promotion is not None),
            float(board.is_castling(move)),
            float(board.is_en_passant(move)),
            dx,
            dy,
        ])

    # Handle positions with no legal moves (checkmate/stalemate)
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(wdl, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_and_sample(csv_path, num_samples=None, random_state=42):
    """
    Load dataset and optionally perform stratified sampling.

    Returns DataFrame with columns: Position, Evaluation (centipawns).
    """
    df = pd.read_csv(csv_path)
    df["cp"] = df["Evaluation"].apply(parse_evaluation)

    if num_samples is not None and num_samples < len(df):
        # Stratified sampling: bin by evaluation range
        bins = [-float("inf"), -5000, -2000, -1000, -500, -200, -100, -50,
                0, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")]
        df["bin"] = pd.cut(df["cp"], bins=bins, labels=False)

        sampled = df.groupby("bin", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), max(1, int(num_samples * len(g) / len(df)))),
                random_state=random_state,
            ),
            include_groups=False,
        )
        # Restore the columns lost by include_groups=False
        sampled = df.loc[sampled.index]

        # If we're still short due to rounding, sample the remainder
        remaining = num_samples - len(sampled)
        if remaining > 0:
            leftover = df.drop(sampled.index)
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=random_state)
            sampled = pd.concat([sampled, extra])

        df = sampled.head(num_samples).reset_index(drop=True)
        df = df.drop(columns=["bin"])

    return df


def build_graphs(df, cache_path=None):
    """
    Convert DataFrame of positions to list of PyG Data objects.
    Uses cache if available.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached graphs from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    graphs = []
    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        fen = row["Position"]
        cp = row["cp"]
        wdl = cp_to_wdl(cp)
        try:
            graph = fen_to_graph(fen, wdl)
            graph.cp = torch.tensor([cp], dtype=torch.float)
            graphs.append(graph)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing FEN '{fen}': {e}")

    if errors > 0:
        print(f"Total errors: {errors}/{len(df)}")

    if cache_path:
        print(f"Saving {len(graphs)} graphs to cache: {cache_path}")
        torch.save(graphs, cache_path)

    return graphs
