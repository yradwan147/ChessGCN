# ChessGCN

A chess engine that evaluates positions and selects moves using Graph Attention Networks (GATv2), trained via supervised learning on Stockfish evaluations and refined through AlphaZero-style self-play with MCTS.

## How It Works

The board is represented as a graph:
- **64 nodes** (one per square) with 21-dimensional features encoding piece type, position, castling rights, en passant, and turn
- **Edges** from legal moves with 10-dimensional features: capture, promotion, castling, en passant, displacement (dx, dy), and promotion piece type (4d one-hot)

The model is a **dual-head ResGATv2** (~307K parameters):
- **Value head**: Predicts Win/Draw/Loss probabilities (WDL classification)
- **Policy head**: Scores each legal move via per-edge MLP

Training pipeline:
1. **Supervised pretraining** on 341K Stockfish-evaluated positions (83% WDL accuracy)
2. **Policy bootstrapping** using Stockfish best moves as targets
3. **Self-play refinement** via MCTS with the neural network guiding search

## Quick Start

```bash
# Setup
conda create -n chessgcn python=3.11
conda activate chessgcn
pip install torch torch_geometric chess pandas numpy scikit-learn flask tqdm matplotlib

# Play against the model
python play.py --checkpoint best_model.pt
# Open http://localhost:5000

# View self-play games
python view_games.py --file selfplay_games.jsonl
# Open http://localhost:5001
```

## Training

### Supervised Training
```bash
python train.py --epochs 100 --patience 20
```

### Self-Play Training
```bash
# Bootstrap policy head from supervised model, then run self-play
python selfplay.py --pretrain-policy --checkpoint best_model.pt

# Or run self-play from an existing dual-head checkpoint
python selfplay.py --checkpoint pretrained_dual_head.pt --iterations 30
```

### Kaggle Notebook
The notebook `chess_gcn_kaggle.ipynb` is self-contained and runs the full pipeline (supervised + self-play) on a T4 GPU within the 12-hour Kaggle session limit.

## Project Structure

```
ChessGCN/
  data.py          # FEN-to-graph conversion, WDL targets, stratified sampling
  model.py         # ResGATv2 with dual value/policy heads
  train.py         # Supervised training loop
  mcts.py          # Monte Carlo Tree Search with PUCT selection
  selfplay.py      # Self-play training pipeline (replay buffer, game gen, eval gating)
  engine.py        # Move evaluation engine
  play.py          # Flask web UI for playing against the model
  view_games.py    # Flask web viewer for self-play game replays
  chess_gcn_kaggle.ipynb  # Self-contained Kaggle notebook
```

## Architecture

```
Input: FEN position
    |
    v
fen_to_graph: 64 nodes (21d) + legal move edges (10d)
    |
    v
ResGATv2: input_proj -> 4x [GATv2Conv -> BN -> GATv2Conv -> BN + residual]
    |
    +-- global_mean_pool -> value_head -> WDL logits (3)
    |
    +-- per-edge MLP -> policy logits (one per legal move)
```

MCTS uses both heads: the policy head provides move priors, and the value head evaluates leaf positions. During self-play, the value head is frozen to preserve supervised accuracy while the policy head improves through search.

## References

- Silver et al. (2018) — AlphaZero: Mastering Chess and Shogi by Self-Play
- Wu (2019) — Accelerating Self-Play Learning in Go (KataGo)
- Czech et al. (2023) — Representation Matters: Improving Chess with WDL and Ply Heads
- Maechler et al. (2024) — AlphaGateau: Enhancing Chess RL with Graph Representation (NeurIPS 2024)
