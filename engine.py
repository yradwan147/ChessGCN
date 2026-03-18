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


class OpeningBook:
    """Simple opening book for interactive play.

    Loads openings from a text file (Name | uci_moves) and builds a
    position lookup table. When the current board matches a known
    book position, returns the next book move.
    """

    def __init__(self, path):
        self.positions = {}  # fen_key -> chess.Move
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue
                uci_strs = parts[1].strip().split()
                board = chess.Board()
                for uci in uci_strs:
                    try:
                        move = chess.Move.from_uci(uci)
                        if move not in board.legal_moves:
                            break
                        # Key: position without move counters (so transpositions match)
                        key = " ".join(board.fen().split()[:4])
                        self.positions[key] = move
                        board.push(move)
                    except ValueError:
                        break

    def lookup(self, board):
        """Return book move for position, or None if not in book."""
        key = " ".join(board.fen().split()[:4])
        move = self.positions.get(key)
        if move and move in board.legal_moves:
            return move
        return None

    def __len__(self):
        return len(self.positions)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_architecture(state_dict):
    """Infer hidden dim, heads, and block count from checkpoint weight shapes."""
    # hidden = output dim of input_proj
    hidden = state_dict["input_proj.0.weight"].shape[0]

    # num_blocks = count distinct block indices
    block_ids = set()
    for k in state_dict:
        if k.startswith("blocks."):
            block_ids.add(int(k.split(".")[1]))
    num_blocks = len(block_ids)

    # heads = hidden / per-head dim, inferred from attention weight shape
    # GATv2Conv att has shape [1, heads, per_head_dim] → heads is dim 1
    att_key = "blocks.0.conv1.att"
    if att_key in state_dict:
        heads = state_dict[att_key].shape[1]
    else:
        heads = 4  # fallback

    return hidden, heads, num_blocks


def load_model(checkpoint_path, device=None, hidden=None, heads=None, num_blocks=None):
    """Load a trained ChessGATv2 model from checkpoint.

    Auto-detects architecture from checkpoint weights if hidden/heads/num_blocks
    are not specified.
    """
    if device is None:
        device = get_device()
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    # Auto-detect architecture from weights
    det_hidden, det_heads, det_blocks = infer_architecture(state_dict)
    hidden = hidden or det_hidden
    heads = heads or det_heads
    num_blocks = num_blocks or det_blocks

    has_policy = any(k.startswith("policy_mlp.") for k in state_dict)
    model = ChessGATv2(hidden=hidden, heads=heads, num_blocks=num_blocks,
                       policy_head=has_policy).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Architecture: hidden={hidden}, heads={heads}, blocks={num_blocks}, "
          f"policy={'yes' if has_policy else 'no'}")
    return model, device


@torch.no_grad()
def evaluate_position(model, device, fen):
    """
    Evaluate a single position. Returns WDL probabilities from the
    perspective of the side to move.
    """
    graph = fen_to_graph(fen)
    batch = Batch.from_data_list([graph]).to(device)
    logits, *_ = model(batch)
    probs = F.softmax(logits, dim=1)[0]
    return {
        "win": probs[0].item(),
        "draw": probs[1].item(),
        "loss": probs[2].item(),
    }


@torch.no_grad()
def get_best_move(model, device, board, top_k=5, opening_book=None):
    """
    Evaluate all legal moves and return the best one plus top-k candidates.

    If an opening_book is provided and the position is in the book,
    returns the book move immediately without running inference.

    Returns:
        best_move: chess.Move
        top_moves: list of dicts with keys: move (SAN), uci, win, draw, loss, value
    """
    # Check opening book first
    if opening_book:
        book_move = opening_book.lookup(board)
        if book_move:
            san = board.san(book_move)
            return book_move, [{
                "move": san, "uci": book_move.uci(),
                "win": 0.5, "draw": 0.5, "loss": 0.0,
                "value": 0.0, "book": True,
            }]

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
    logits, *_ = model(batch)
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
