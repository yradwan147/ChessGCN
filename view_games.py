"""
Web viewer for self-play game records.

Loads a JSONL file produced by selfplay.py and lets you step through
each game move by move, showing the board and top-5 MCTS candidates.

Usage:
    python view_games.py                              # default: selfplay_games.jsonl
    python view_games.py --file my_games.jsonl        # custom file
    python view_games.py --port 5001                  # custom port
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify

app = Flask(__name__)

games = []


def load_games(path):
    global games
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    print(f"Loaded {len(games)} games from {path}")


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/games")
def api_games():
    summary = []
    for i, g in enumerate(games):
        summary.append({
            "index": i,
            "iteration": g.get("iteration", "?"),
            "game": g.get("game", "?"),
            "result": g.get("result", "?"),
            "num_moves": g.get("num_moves", len(g.get("moves", []))),
        })
    return jsonify(summary)


@app.route("/api/game/<int:idx>")
def api_game(idx):
    if idx < 0 or idx >= len(games):
        return jsonify({"error": "Game not found"}), 404
    return jsonify(games[idx])


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Self-Play Game Viewer</title>
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
    padding: 20px;
}
.container {
    display: flex;
    gap: 24px;
    max-width: 1200px;
    width: 100%;
    align-items: flex-start;
}
.left-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
#board { width: 480px; height: 480px; }
.controls {
    display: flex;
    gap: 8px;
    justify-content: center;
    align-items: center;
}
.controls button {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a5276;
}
.controls button:hover { background: #1a5276; }
.controls button:disabled { opacity: 0.3; cursor: default; }
#move-counter {
    font-family: 'Courier New', monospace;
    min-width: 80px;
    text-align: center;
}
.sidebar {
    flex: 1;
    min-width: 280px;
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
#game-list {
    max-height: 180px;
    overflow-y: auto;
}
.game-item {
    padding: 6px 10px;
    cursor: pointer;
    border-radius: 4px;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    font-family: 'Courier New', monospace;
}
.game-item:hover { background: rgba(255,255,255,0.05); }
.game-item.active { background: rgba(15,52,96,0.8); border: 1px solid #1a5276; }
.game-item .result { font-weight: 700; }
.result-w { color: #4caf50; }
.result-b { color: #ef5350; }
.result-d { color: #888; }
#game-info {
    font-size: 13px;
    color: #aaa;
    line-height: 1.6;
}
.top-moves-list { display: flex; flex-direction: column; gap: 6px; }
.top-move {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    padding: 4px 6px;
    border-radius: 4px;
}
.top-move.chosen { background: rgba(46,125,50,0.2); }
.top-move .move-name {
    font-family: 'Courier New', monospace;
    font-weight: 700;
    min-width: 55px;
}
.top-move .bar-container {
    flex: 1;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    background: #333;
    position: relative;
}
.top-move .visit-bar {
    height: 100%;
    background: #1a5276;
    border-radius: 3px;
}
.top-move .stats {
    min-width: 100px;
    text-align: right;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #aaa;
}
#move-list-panel {
    max-height: 120px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    color: #aaa;
}
.move-link { cursor: pointer; padding: 1px 3px; border-radius: 3px; }
.move-link:hover { background: rgba(255,255,255,0.1); }
.move-link.current { background: rgba(46,125,50,0.3); color: #fff; }
</style>
</head>
<body>
<div class="container">
    <div class="left-panel">
        <div id="board"></div>
        <div class="controls">
            <button id="btn-start" onclick="goToMove(0)">&laquo;</button>
            <button id="btn-prev" onclick="goToMove(currentMove - 1)">&lsaquo;</button>
            <span id="move-counter">0 / 0</span>
            <button id="btn-next" onclick="goToMove(currentMove + 1)">&rsaquo;</button>
            <button id="btn-end" onclick="goToMove(totalMoves)">&raquo;</button>
        </div>
    </div>
    <div class="sidebar">
        <div class="panel">
            <h3>Games</h3>
            <div id="game-list">Loading...</div>
        </div>
        <div class="panel">
            <h3>Game Info</h3>
            <div id="game-info">Select a game</div>
        </div>
        <div class="panel">
            <h3>Moves</h3>
            <div id="move-list-panel"></div>
        </div>
        <div class="panel">
            <h3>MCTS Top 5 Candidates</h3>
            <div id="top-moves" class="top-moves-list">
                <span style="color:#666;font-size:13px;">Select a game to view</span>
            </div>
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
let boardObj = null;
let currentGame = null;
let currentMove = 0;
let totalMoves = 0;

function initBoard() {
    boardObj = Chessboard('board', {
        position: 'start',
        draggable: false,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
    });
}

function loadGameList() {
    $.get('/api/games', function(data) {
        let html = '';
        data.forEach(function(g) {
            let rcls = g.result === '1-0' ? 'result-w' : (g.result === '0-1' ? 'result-b' : 'result-d');
            html += '<div class="game-item" data-idx="' + g.index + '" onclick="loadGame(' + g.index + ')">' +
                '<span>Iter ' + g.iteration + ' #' + g.game + ' (' + g.num_moves + ' moves)</span>' +
                '<span class="result ' + rcls + '">' + g.result + '</span>' +
                '</div>';
        });
        if (!html) html = '<span style="color:#666">No games found</span>';
        $('#game-list').html(html);
    });
}

function loadGame(idx) {
    $.get('/api/game/' + idx, function(data) {
        currentGame = data;
        currentMove = 0;
        totalMoves = data.moves.length;

        // Highlight active game
        $('.game-item').removeClass('active');
        $('.game-item[data-idx="' + idx + '"]').addClass('active');

        // Game info
        $('#game-info').html(
            'Iteration: ' + data.iteration + '<br>' +
            'Game: ' + data.game + '<br>' +
            'Result: <strong>' + data.result + '</strong><br>' +
            'Moves: ' + data.num_moves
        );

        // Build move list
        buildMoveList();
        goToMove(0);
    });
}

function buildMoveList() {
    if (!currentGame) return;
    let html = '';
    let moves = currentGame.moves;
    for (let i = 0; i < moves.length; i += 2) {
        let num = Math.floor(i / 2) + 1;
        let san1 = moves[i].move_san || moves[i].move;
        html += '<span class="move-link" data-move="' + (i + 1) + '" onclick="goToMove(' + (i + 1) + ')">' +
                num + '. ' + san1 + '</span> ';
        if (i + 1 < moves.length) {
            let san2 = moves[i + 1].move_san || moves[i + 1].move;
            html += '<span class="move-link" data-move="' + (i + 2) + '" onclick="goToMove(' + (i + 2) + ')">' +
                    san2 + '</span>  ';
        }
    }
    $('#move-list-panel').html(html);
}

function goToMove(n) {
    if (!currentGame) return;
    n = Math.max(0, Math.min(n, totalMoves));
    currentMove = n;

    // Show the FEN at this move (move N means after N moves have been played)
    let fen;
    if (n === 0) {
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    } else {
        // The FEN at move[n-1] is BEFORE that move was played,
        // so we need the FEN at move[n] or the start position after pushing moves.
        // Actually, each entry stores the FEN *before* the move.
        // So move[n] FEN = position before move n+1 = position after move n.
        if (n < totalMoves) {
            fen = currentGame.moves[n].fen;
        } else {
            // After last move: we don't have a FEN stored, reconstruct
            // by noting the last entry's FEN is before the last move
            // We'll just use the last stored FEN for simplicity
            // In practice, the viewer shows up to the last move's candidates
            fen = currentGame.moves[n - 1].fen;
        }
    }
    boardObj.position(fen, true);

    // Update counter
    $('#move-counter').text(n + ' / ' + totalMoves);

    // Update button states
    $('#btn-start, #btn-prev').prop('disabled', n === 0);
    $('#btn-next, #btn-end').prop('disabled', n >= totalMoves);

    // Highlight current move in move list
    $('.move-link').removeClass('current');
    if (n > 0) {
        $('.move-link[data-move="' + n + '"]').addClass('current');
    }

    // Show top-5 for the current position (before the move at index n)
    // When n=0, show candidates for move 0 (white's first move)
    // When n>0, show candidates for the move that was just played (index n-1)
    let moveIdx = (n > 0) ? n - 1 : 0;
    if (n === 0 && totalMoves > 0) {
        showTop5(currentGame.moves[0], currentGame.moves[0].move);
    } else if (n > 0 && n <= totalMoves) {
        let entry = currentGame.moves[n - 1];
        showTop5(entry, entry.move);
    } else {
        $('#top-moves').html('<span style="color:#666">No data</span>');
    }
}

function showTop5(entry, chosenUci) {
    let tm = $('#top-moves');
    tm.empty();
    if (!entry || !entry.top5 || entry.top5.length === 0) {
        tm.html('<span style="color:#666">No candidates</span>');
        return;
    }

    let maxVisits = Math.max(...entry.top5.map(m => m.visits));

    entry.top5.forEach(function(m) {
        let isChosen = m.uci === chosenUci;
        let cls = isChosen ? 'top-move chosen' : 'top-move';
        let barW = maxVisits > 0 ? (m.visits / maxVisits * 100) : 0;
        let qStr = m.q >= 0 ? '+' + m.q.toFixed(3) : m.q.toFixed(3);
        let html = '<div class="' + cls + '">' +
            '<span class="move-name">' + m.san + '</span>' +
            '<div class="bar-container"><div class="visit-bar" style="width:' + barW + '%"></div></div>' +
            '<span class="stats">N=' + m.visits + ' Q=' + qStr + '</span>' +
            '</div>';
        tm.append(html);
    });

    // Root Q
    if (entry.root_q !== undefined) {
        let rq = entry.root_q >= 0 ? '+' + entry.root_q.toFixed(3) : entry.root_q.toFixed(3);
        tm.append('<div style="font-size:11px;color:#666;margin-top:6px;">Root Q: ' + rq +
                  ' | Side: ' + entry.side + '</div>');
    }
}

// Keyboard navigation
$(document).keydown(function(e) {
    if (e.key === 'ArrowLeft') { goToMove(currentMove - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { goToMove(currentMove + 1); e.preventDefault(); }
    if (e.key === 'Home') { goToMove(0); e.preventDefault(); }
    if (e.key === 'End') { goToMove(totalMoves); e.preventDefault(); }
});

$(document).ready(function() {
    initBoard();
    loadGameList();
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="View self-play game records")
    parser.add_argument("--file", type=str, default="selfplay_games.jsonl",
                        help="Path to JSONL game records file")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found. Run selfplay.py first to generate games.")
        return

    load_games(path)
    print(f"\nOpen http://{args.host}:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
