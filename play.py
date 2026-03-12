"""
Web interface for playing against or watching the Chess GCN model.

Usage:
    python play.py                           # default: best_model.pt
    python play.py --checkpoint my_model.pt  # custom checkpoint
    python play.py --port 8080               # custom port
"""

import argparse

import chess
from flask import Flask, jsonify, request

from engine import load_model, get_best_move, evaluate_position

app = Flask(__name__)

# Global state
board = chess.Board()
model = None
device = None
move_history = []
player_color = chess.WHITE  # which side the human plays
auto_play = False
top_moves_cache = []


def game_status():
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"checkmate — {winner} wins"
    if board.is_stalemate():
        return "stalemate — draw"
    if board.is_insufficient_material():
        return "insufficient material — draw"
    if board.is_fifty_moves():
        return "50-move rule — draw"
    if board.is_repetition(3):
        return "threefold repetition — draw"
    if board.is_check():
        return "check"
    return "playing"


def board_state(top_moves=None):
    status = game_status()
    ev = evaluate_position(model, device, board.fen())
    return {
        "fen": board.fen(),
        "moves": move_history,
        "status": status,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "player_color": "white" if player_color == chess.WHITE else "black",
        "eval": ev,
        "top_moves": top_moves or top_moves_cache,
        "is_game_over": board.is_game_over(),
    }


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/state")
def api_state():
    return jsonify(board_state())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    global board, move_history, player_color, auto_play, top_moves_cache
    data = request.get_json(force=True)
    color = data.get("color", "white")

    board = chess.Board()
    move_history = []
    top_moves_cache = []

    if color == "auto":
        auto_play = True
        player_color = chess.WHITE
    else:
        auto_play = False
        player_color = chess.WHITE if color == "white" else chess.BLACK

    # If model plays white (human is black), make the first move
    result = board_state()
    if not auto_play and player_color == chess.BLACK:
        best, top = get_best_move(model, device, board)
        if best:
            san = board.san(best)
            board.push(best)
            move_history.append(san)
            top_moves_cache = top
            result = board_state(top)

    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def api_move():
    """Handle a human move, then respond with the model's move."""
    global top_moves_cache
    data = request.get_json(force=True)
    from_sq = data.get("from")
    to_sq = data.get("to")
    promotion = data.get("promotion")

    uci_str = from_sq + to_sq
    if promotion:
        uci_str += promotion

    try:
        move = chess.Move.from_uci(uci_str)
    except (ValueError, chess.InvalidMoveError):
        return jsonify({"error": "Invalid move format"}), 400

    if move not in board.legal_moves:
        # Try with promotion
        move = chess.Move.from_uci(uci_str + "q")
        if move not in board.legal_moves:
            return jsonify({"error": "Illegal move"}), 400

    # Apply human's move
    san = board.san(move)
    board.push(move)
    move_history.append(san)

    # Model responds if game is not over
    top = []
    if not board.is_game_over():
        best, top = get_best_move(model, device, board)
        if best:
            san = board.san(best)
            board.push(best)
            move_history.append(san)

    top_moves_cache = top
    return jsonify(board_state(top))


@app.route("/api/auto_move", methods=["POST"])
def api_auto_move():
    """Model makes a single move (for auto-play mode)."""
    global top_moves_cache
    if board.is_game_over():
        return jsonify(board_state())

    best, top = get_best_move(model, device, board)
    if best:
        san = board.san(best)
        board.push(best)
        move_history.append(san)

    top_moves_cache = top
    return jsonify(board_state(top))


# ── HTML Template ────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chess GCN</title>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
}
.container {
    display: flex;
    gap: 24px;
    max-width: 1100px;
    width: 100%;
    align-items: flex-start;
}
.board-section {
    display: flex;
    gap: 12px;
    align-items: stretch;
}
#board {
    width: 480px;
    height: 480px;
}
.eval-bar-container {
    width: 28px;
    background: #c62828;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
    height: 480px;
}
.eval-bar-win {
    background: #2e7d32;
    width: 100%;
    position: absolute;
    top: 0;
    transition: height 0.4s ease;
}
.eval-bar-draw {
    background: #757575;
    width: 100%;
    position: absolute;
    transition: top 0.4s ease, height 0.4s ease;
}
.sidebar {
    flex: 1;
    min-width: 260px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.panel {
    background: #16213e;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #0f3460;
}
.panel h3 {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 10px;
}
.buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.buttons button {
    padding: 8px 14px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    transition: background 0.2s;
}
.btn-white { background: #e0e0e0; color: #1a1a2e; }
.btn-black { background: #444; color: #e0e0e0; }
.btn-auto { background: #0f3460; color: #e0e0e0; border: 1px solid #1a5276 !important; }
.btn-white:hover { background: #fff; }
.btn-black:hover { background: #666; }
.btn-auto:hover { background: #1a5276; }
#status {
    font-size: 15px;
    font-weight: 600;
    padding: 8px 0;
    min-height: 30px;
}
#move-list {
    max-height: 140px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: #aaa;
}
.top-moves-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.top-move {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    padding: 4px 6px;
    border-radius: 4px;
}
.top-move.best { background: rgba(46,125,50,0.2); }
.top-move .move-name {
    font-family: 'Courier New', monospace;
    font-weight: 700;
    min-width: 50px;
}
.top-move .wdl-bar {
    flex: 1;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    display: flex;
}
.wdl-w { background: #2e7d32; }
.wdl-d { background: #757575; }
.wdl-l { background: #c62828; }
.top-move .value {
    min-width: 45px;
    text-align: right;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #aaa;
}
.eval-text {
    font-size: 12px;
    color: #888;
    text-align: center;
    margin-top: 4px;
}
</style>
</head>
<body>
<div class="container">
    <div class="board-section">
        <div class="eval-bar-container">
            <div class="eval-bar-win" id="eval-win"></div>
            <div class="eval-bar-draw" id="eval-draw"></div>
        </div>
        <div id="board"></div>
    </div>
    <div class="sidebar">
        <div class="panel">
            <h3>New Game</h3>
            <div class="buttons">
                <button class="btn-white" onclick="newGame('white')">Play White</button>
                <button class="btn-black" onclick="newGame('black')">Play Black</button>
                <button class="btn-auto" onclick="newGame('auto')">Auto-Play</button>
            </div>
        </div>
        <div class="panel">
            <h3>Status</h3>
            <div id="status">Click a button to start</div>
        </div>
        <div class="panel">
            <h3>Moves</h3>
            <div id="move-list"></div>
        </div>
        <div class="panel">
            <h3>Model's Top Moves</h3>
            <div id="top-moves" class="top-moves-list">
                <span style="color:#666;font-size:13px;">Waiting for model move...</span>
            </div>
            <div class="eval-text" id="eval-text"></div>
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
let boardObj = null;
let gameState = null;
let playerColor = 'white';
let isAutoPlay = false;
let autoPlayTimer = null;

function initBoard() {
    boardObj = Chessboard('board', {
        position: 'start',
        draggable: true,
        pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/img/chesspieces/wikipedia/{piece}.png',
        onDrop: onDrop,
        onDragStart: onDragStart,
    });
}

function onDragStart(source, piece) {
    if (isAutoPlay) return false;
    if (gameState && gameState.is_game_over) return false;
    // Only allow dragging own pieces
    if (playerColor === 'white' && piece.search(/^b/) !== -1) return false;
    if (playerColor === 'black' && piece.search(/^w/) !== -1) return false;
    // Only on player's turn
    if (gameState && gameState.turn !== playerColor) return false;
    return true;
}

function onDrop(source, target, piece) {
    // Detect pawn promotion
    let promotion = null;
    if (piece === 'wP' && target[1] === '8') promotion = 'q';
    if (piece === 'bP' && target[1] === '1') promotion = 'q';

    $.ajax({
        url: '/api/move',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({from: source, to: target, promotion: promotion}),
        success: function(data) {
            updateUI(data);
        },
        error: function() {
            boardObj.position(gameState ? gameState.fen : 'start');
        }
    });
    return undefined; // don't snap back yet, wait for server
}

function newGame(color) {
    stopAutoPlay();
    isAutoPlay = (color === 'auto');
    playerColor = (color === 'black') ? 'black' : 'white';

    $.ajax({
        url: '/api/new_game',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({color: color}),
        success: function(data) {
            boardObj.orientation(isAutoPlay ? 'white' : playerColor);
            updateUI(data);
            if (isAutoPlay) startAutoPlay();
        }
    });
}

function startAutoPlay() {
    autoPlayTimer = setInterval(function() {
        if (gameState && gameState.is_game_over) {
            stopAutoPlay();
            return;
        }
        $.ajax({
            url: '/api/auto_move',
            method: 'POST',
            contentType: 'application/json',
            data: '{}',
            success: function(data) {
                updateUI(data);
                if (data.is_game_over) stopAutoPlay();
            }
        });
    }, 1200);
}

function stopAutoPlay() {
    if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
    }
}

function updateUI(data) {
    gameState = data;
    boardObj.position(data.fen, true);

    // Status
    let statusText = data.status;
    if (data.status === 'playing') {
        if (isAutoPlay) {
            statusText = data.turn + ' to move (auto-play)';
        } else {
            statusText = data.turn === playerColor ? 'Your turn' : 'Model thinking...';
        }
    }
    $('#status').text(statusText.charAt(0).toUpperCase() + statusText.slice(1));

    // Eval bar
    if (data.eval) {
        let w = data.eval.win * 100;
        let d = data.eval.draw * 100;
        $('#eval-win').css('height', w + '%');
        $('#eval-draw').css({top: w + '%', height: d + '%'});
        let val = (data.eval.win - data.eval.loss);
        $('#eval-text').text('W:' + (data.eval.win*100).toFixed(0) + '% D:' + (data.eval.draw*100).toFixed(0) + '% L:' + (data.eval.loss*100).toFixed(0) + '%');
    }

    // Move list
    let ml = '';
    for (let i = 0; i < data.moves.length; i += 2) {
        let num = Math.floor(i/2) + 1;
        ml += num + '. ' + data.moves[i];
        if (i+1 < data.moves.length) ml += ' ' + data.moves[i+1];
        ml += '  ';
    }
    $('#move-list').text(ml);

    // Top moves
    let tm = $('#top-moves');
    tm.empty();
    if (data.top_moves && data.top_moves.length > 0) {
        data.top_moves.forEach(function(m, idx) {
            let cls = idx === 0 ? 'top-move best' : 'top-move';
            let wp = (m.win * 100).toFixed(0);
            let dp = (m.draw * 100).toFixed(0);
            let lp = (m.loss * 100).toFixed(0);
            let html = '<div class="' + cls + '">' +
                '<span class="move-name">' + m.move + '</span>' +
                '<div class="wdl-bar">' +
                    '<div class="wdl-w" style="width:' + wp + '%"></div>' +
                    '<div class="wdl-d" style="width:' + dp + '%"></div>' +
                    '<div class="wdl-l" style="width:' + lp + '%"></div>' +
                '</div>' +
                '<span class="value">' + (m.value >= 0 ? '+' : '') + m.value.toFixed(2) + '</span>' +
                '</div>';
            tm.append(html);
        });
    } else {
        tm.html('<span style="color:#666;font-size:13px;">Waiting for model move...</span>');
    }
}

$(document).ready(function() {
    initBoard();
    $.get('/api/state', function(data) { updateUI(data); });
});
</script>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global model, device

    parser = argparse.ArgumentParser(description="Play against the Chess GCN")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print("Loading model...")
    model, device = load_model(args.checkpoint)
    print(f"Model loaded on {device}")
    print(f"\nOpen http://{args.host}:{args.port} in your browser\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
