# Research-Driven Improvements (Two Phases)

## Context
Deep analysis of AlphaGateau, KataGo, Lc0, Czech et al., and DeepMind Searchless Chess reveals concrete gaps. Two phases of changes, all runs in parallel within each phase.

---

## Phase 1: Data + Training Process (no architecture changes)

Changes to `data.py` and `selfplay.py` only. Model stays the same.

### P1-1: Self-edges in GNN
**File:** `data.py`, `fen_to_graph()`
AlphaGateau explicitly adds self-edges. Without them, nodes lose identity after 8 GATv2 layers.
- Add 64 edges: `(i, i)` for `i in 0..63`
- Edge features: existing 10d + 1d self-edge flag = **11d total** (EDGE_DIM changes to 11)
- Self-edges use `[0,0,0,0,0,0, 0,0,0,0, 1]` (all zeros + self flag)
- **Policy head must filter self-edges**: in `model.py` forward, mask out edges where `edge_index[0] == edge_index[1]` before policy MLP

### P1-2: Is-in-check feature
**File:** `data.py`, `fen_to_graph()`
Add 1d binary: 1 if side-to-move is in check. Node features become **22d**.
- `check_flag = 1.0 if board.is_check() else 0.0`
- Append to every node's feature vector

### P1-3: WDL_K constant: 111 → 200
**File:** `data.py`, `WDL_K` constant
Our K=111 creates a very narrow draw class. Stockfish's calibrated K≈271.6. K=200 gives a wider, less noisy draw region.

### P1-4: EMA weight averaging
**File:** `selfplay.py`
After each `optimizer.step()`, update EMA shadow model. Use EMA model for self-play game generation and evaluation. Training model stays as-is.
```python
from copy import deepcopy

# Setup (after model creation):
ema_model = deepcopy(model)
ema_decay = 0.995

# After each optimizer.step():
with torch.no_grad():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

# Use ema_model (not model) for MCTS game generation and eval
```

### P1-5: Games/iter: 3 → 30
**Effort: zero code changes, CLI flag only.**
30 games × ~80 pos = 2400 pos/iter (10x current). With Gumbel 32 sims, ~2 min/game = ~60 min self-play per iter. 40 iters ≈ 40h total.

### P1-6: Color flipping augmentation
**File:** `data.py` or `selfplay.py` training loop
Use `chess.Board.mirror()` to flip colors, then re-generate graph via `fen_to_graph()`. Apply 50% randomly during training (not game generation).
```python
import chess

def augment_color_flip(fen, value_target):
    """Flip colors: swap White/Black, mirror board vertically, swap W/L."""
    board = chess.Board(fen)
    mirrored = board.mirror()
    new_fen = mirrored.fen()
    new_value = -value_target  # Flip perspective
    return new_fen, new_value
```
In `train_on_buffer()`: for each sampled position, 50% chance apply flip before creating the graph. The policy target stays valid because `fen_to_graph()` regenerates legal moves for the mirrored position — but the policy target from the original position doesn't map to the mirrored legal moves. **Therefore: color flipping only applies to the value loss, not policy loss.** Split the batch: original positions train both heads, flipped positions train value head only.

Actually, simpler approach: apply flipping during `add_game_to_buffer()`. For each position added, also add the mirror with flipped value. This doubles the buffer but uses correct policy targets for original and correct value targets for flipped (no policy target for flipped).

### P1-7: Playout cap randomization
**File:** `selfplay.py`, `play_one_game()`
KataGo technique: 75% of moves use fast search, 25% full search.
- **Fast search**: 8 sims, no Dirichlet noise, no policy target recorded. Value target still used.
- **Full search**: 32 sims, with noise, policy target recorded normally.
- In `game_data`: mark each position with `is_full_search` flag
- In `add_game_to_buffer()`: only add policy target for full-search positions; for fast-search positions, set policy_target to None/zeros (or skip them from policy loss)
- Average visits per move: 0.75×8 + 0.25×32 = 14 → ~2.3x more moves per compute than 32 everywhere
- CLI: `--playout-cap` flag to enable, `--fast-sims 8` for fast search count

### Phase 1 Sweep (12 runs, 40 iters, all parallel):

| Run | Self-edges | Check | WDL_K | EMA | Games | Color flip | Playout cap |
|-----|-----------|-------|-------|-----|-------|------------|-------------|
| p1_baseline | — | — | 111 | — | 3 | — | — |
| p2_selfedge | **Yes** | — | 111 | — | 3 | — | — |
| p3_check | — | **Yes** | 111 | — | 3 | — | — |
| p4_wdlk | — | — | **200** | — | 3 | — | — |
| p5_ema | — | — | 111 | **Yes** | 3 | — | — |
| p6_30games | — | — | 111 | — | **30** | — | — |
| p7_colorflip | — | — | 111 | — | 3 | **Yes** | — |
| p8_playoutcap | — | — | 111 | — | 3 | — | **Yes** |
| p9_features | **Yes** | **Yes** | **200** | — | 3 | — | — |
| p10_data | — | — | 111 | — | **30** | **Yes** | **Yes** |
| p11_training | — | — | 111 | **Yes** | 3 | — | **Yes** |
| p12_all_p1 | **Yes** | **Yes** | **200** | **Yes** | **30** | **Yes** | **Yes** |

- p1-p8: isolate each change
- p9: all feature/data.py changes combined
- p10: all data volume changes combined
- p11: all training process changes combined
- p12: everything

---

## Phase 2: Architecture Changes

Changes to `model.py` and `selfplay.py`. Run in parallel with Phase 1.

### P2-1: Global attention pooling
**File:** `model.py`
Replace `global_mean_pool` with PyG's `GlobalAttention` in value head. AlphaGateau uses this.
```python
from torch_geometric.nn import GlobalAttention

# In ChessGATv2.__init__:
gate_nn = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
self.attn_pool = GlobalAttention(gate_nn)

# In forward:
x_pool = self.attn_pool(x, batch)  # replaces global_mean_pool(x, batch)
```

### P2-2: Moves-left auxiliary head
**File:** `model.py`, `selfplay.py`
Already partially implemented in notebook. Czech et al. (2024): +33 Elo.
- Architecture: `Linear(hidden→64) → ReLU → Linear(64→1)`
- Loss: `0.002 * MSE(predicted, target)` where target = `(total_moves - current_move) / max_moves`
- Forward returns 3 values: `(value_logits, policy_logits, moves_left)`

### P2-3: Score distribution head (128 bins)
**File:** `model.py`, `data.py`, `selfplay.py`
Replace 3-class WDL with 128-bin distribution over win probability.
- **Bins**: 128 uniformly spaced over [0, 1] win probability
- **Target encoding**: HL-Gauss — soft Gaussian centered on true win%, sigma = 0.75 × bin_width
  ```python
  bin_centers = torch.linspace(0, 1, 128)
  bin_width = bin_centers[1] - bin_centers[0]
  sigma = 0.75 * bin_width
  target = torch.exp(-0.5 * ((bin_centers - true_win_pct) / sigma) ** 2)
  target = target / target.sum()  # normalize to distribution
  ```
- **Loss**: Cross-entropy: `L = -sum(target_i * log(pred_i))`
- **Inference**: Expected value = `sum(softmax(logits) * bin_centers)`, then `value = 2 * expected - 1` to get [-1, 1] range for MCTS
- **Converting cp to win%**: `win_pct = 1.0 / (1.0 + exp(-cp / WDL_K))`
- Replaces value_head output from (batch, 3) to (batch, 128)
- Policy head unchanged

### Phase 2 Sweep (6 runs, 40 iters, all parallel):

| Run | Attn pool | Moves-left | Score dist (128 bins) |
|-----|-----------|------------|----------------------|
| a1_baseline | — | — | — |
| a2_attnpool | **Yes** | — | — |
| a3_movesleft | — | **Yes** | — |
| a4_scoredist | — | — | **Yes** |
| a5_arch_combo | **Yes** | **Yes** | — |
| a6_all_p2 | **Yes** | **Yes** | **Yes** |

---

## Files to Modify

### Phase 1:
| File | Changes |
|------|---------|
| `data.py` | Self-edges (EDGE_DIM→11), is-in-check (22d nodes), WDL_K=200, color flip helper |
| `selfplay.py` | EMA model, playout cap logic, color flip in buffer, CLI flags |
| `model.py` | Filter self-edges in policy head |
| `slurm/submit_research_p1.sh` | 12-run sweep |

### Phase 2:
| File | Changes |
|------|---------|
| `model.py` | GlobalAttention pooling, moves-left head, 128-bin score dist head |
| `selfplay.py` | Moves-left loss, score dist loss/targets, CLI flags |
| `data.py` | cp_to_win_pct conversion for score dist targets |
| `slurm/submit_research_p2.sh` | 6-run sweep |

### New CLI flags:
```
# Phase 1
--no-self-edges          # disable self-edges (on by default)
--no-check-feature       # disable is-in-check (on by default)
--wdl-k 200              # WDL_K constant (default 200)
--use-ema                # enable EMA weight averaging
--color-flip             # enable color flip augmentation
--playout-cap            # enable playout cap randomization
--fast-sims 8            # sims for fast search in playout cap

# Phase 2
--attn-pool              # enable attention pooling
--moves-left-head        # enable moves-left auxiliary head
--score-dist-bins 128    # enable score distribution head (0 = use WDL)
```

## Verification
- Self-edges: graph should have 64+N_moves edges, 11d features
- Check feature: 22d nodes, value=1 in check positions
- WDL_K: `cp_to_wdl(50)` should show more draw probability
- EMA: ema_model params differ slightly from model after training
- 30 games: buffer has ~2400 entries after 1 iter
- Color flip: mirrored FEN is valid, value target is negated
- Playout cap: ~75% of moves have 8 sims, ~25% have 32
- Attention pool: value head uses GlobalAttention, policy head unchanged
- Score dist: value output shape (batch, 128), loss is cross-entropy
- Moves-left: forward returns 3 values, MSE loss on normalized target

## Future (save to memory only)
- **Soft policy targets from Stockfish multi-PV** — requires dataset reprocessing
- **Opponent move prediction** — ChessCoach proved it HURTS in chess, skip entirely
