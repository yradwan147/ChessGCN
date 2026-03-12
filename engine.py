"""
Chess engine powered by the trained GCN model.
Evaluates positions and selects moves via batched inference.
"""

import chess
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from data import fen_to_graph
from model import ChessGATv2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device=None):
    """Load a trained ChessGATv2 model from checkpoint."""
    if device is None:
        device = get_device()
    model = ChessGATv2().to(device)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@torch.no_grad()
def evaluate_position(model, device, fen):
    """
    Evaluate a single position. Returns WDL probabilities from the
    perspective of the side to move.
    """
    graph = fen_to_graph(fen)
    batch = Batch.from_data_list([graph]).to(device)
    logits, _ = model(batch)
    probs = F.softmax(logits, dim=1)[0]
    return {
        "win": probs[0].item(),
        "draw": probs[1].item(),
        "loss": probs[2].item(),
    }


@torch.no_grad()
def get_best_move(model, device, board, top_k=5):
    """
    Evaluate all legal moves and return the best one plus top-k candidates.

    For each legal move:
      - Apply it to a copy of the board
      - Evaluate the resulting position (from the opponent's perspective)
      - Negate the value (opponent's win = our loss)

    Returns:
        best_move: chess.Move
        top_moves: list of dicts with keys: move (SAN), uci, win, draw, loss, value
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, []

    # Build graphs for all resulting positions
    graphs = []
    for move in legal_moves:
        b = board.copy()
        b.push(move)
        graph = fen_to_graph(b.fen())
        graphs.append(graph)

    # Batch inference
    batch = Batch.from_data_list(graphs).to(device)
    logits, _ = model(batch)
    probs = F.softmax(logits, dim=1)  # [num_moves, 3]

    # Value from opponent's perspective, negated = value for us
    # probs[:, 0] = opponent's P(win), probs[:, 2] = opponent's P(loss)
    # Our value = -(opponent's win - opponent's loss) = opponent's loss - opponent's win
    our_values = (probs[:, 2] - probs[:, 0]).cpu().tolist()

    # Build results
    results = []
    for i, move in enumerate(legal_moves):
        san = board.san(move)
        # WDL from OUR perspective (flip opponent's W and L)
        results.append({
            "move": san,
            "uci": move.uci(),
            "win": probs[i, 2].item(),   # opponent's loss = our win
            "draw": probs[i, 1].item(),
            "loss": probs[i, 0].item(),   # opponent's win = our loss
            "value": our_values[i],
        })

    # Sort by value descending (best for us first)
    results.sort(key=lambda x: x["value"], reverse=True)

    best_move_uci = results[0]["uci"]
    best_move = chess.Move.from_uci(best_move_uci)

    return best_move, results[:top_k]
