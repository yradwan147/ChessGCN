#!/bin/bash
# ============================================================
# Anti-Collapse Sweep: fix self-play catastrophic forgetting
# ============================================================
#
# Architecture locked: 192h, 4blk, 1head
# Base config: Gumbel 32 sims, frozen value, 52% gate, eval every 5
# All runs: 60 iterations, 3 games/iter, 200 train steps
#
# Tests: sliding window buffer, weighted replay, checkpoint restoration
#
# Usage: bash slurm/submit_collapse_sweep.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32

# Shared base flags
BASE="--freeze-value --iterations 60 --eval-interval 5 --gate-threshold 0.52"

echo "============================================================"
echo "ChessGCN Anti-Collapse Sweep (6 runs)"
echo "Architecture: ${H}h ${BLK}blk ${HEADS}head (locked)"
echo "Base: Gumbel, frozen, 52% gate, eval@5, 60 iters"
echo "============================================================"
echo ""

# ── c1: Control (no anti-collapse, reproduces td_aggr_gate) ──
echo "  c1: Control (no fixes)"
sbatch -J c1_control slurm/run_experiment.sh c1_control $H $HEADS $BLK $SIMS 3 \
    $BASE
COUNT=$((COUNT + 1))

# ── c2: Sliding window, 5 iterations ──
echo "  c2: Window 5 iters"
sbatch -J c2_window5 slurm/run_experiment.sh c2_window5 $H $HEADS $BLK $SIMS 3 \
    $BASE --buffer-window 5
COUNT=$((COUNT + 1))

# ── c3: Sliding window, 10 iterations ──
echo "  c3: Window 10 iters"
sbatch -J c3_window10 slurm/run_experiment.sh c3_window10 $H $HEADS $BLK $SIMS 3 \
    $BASE --buffer-window 10
COUNT=$((COUNT + 1))

# ── c4: Weighted replay, decay 0.9 ──
echo "  c4: Weighted replay (decay 0.9)"
sbatch -J c4_weighted slurm/run_experiment.sh c4_weighted $H $HEADS $BLK $SIMS 3 \
    $BASE --replay-decay 0.9
COUNT=$((COUNT + 1))

# ── c5: Restore on collapse ──
echo "  c5: Restore on collapse"
sbatch -J c5_restore slurm/run_experiment.sh c5_restore $H $HEADS $BLK $SIMS 3 \
    $BASE --restore-on-collapse
COUNT=$((COUNT + 1))

# ── c6: Window 10 + restore (combined) ──
echo "  c6: Window 10 + restore (combined)"
sbatch -J c6_combined slurm/run_experiment.sh c6_combined $H $HEADS $BLK $SIMS 3 \
    $BASE --buffer-window 10 --restore-on-collapse
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs"
echo "============================================================"
echo ""
echo "  c1_control   — No fixes (baseline, reproduces td_aggr_gate)"
echo "  c2_window5   — Only train on last 5 iters of data"
echo "  c3_window10  — Only train on last 10 iters of data"
echo "  c4_weighted  — Exponential decay weighting (0.9^age)"
echo "  c5_restore   — Restore best model if eval < 35% twice"
echo "  c6_combined  — Window 10 + restore"
echo ""
echo "Comparisons:"
echo "  c1 vs c2/c3:  Does windowing prevent collapse?"
echo "  c2 vs c3:     Tight (5) vs loose (10) window"
echo "  c1 vs c4:     Does soft weighting help vs hard window?"
echo "  c1 vs c5:     Does restoration recover from collapse?"
echo "  c6 vs c3/c5:  Is combining fixes better than each alone?"
echo "============================================================"
echo ""
echo "Estimated: ~5-7h per run, all within 30h limit"
echo ""
echo "Monitor:"
echo "  for d in results/c*/; do printf '%-20s ' \"\$d\"; grep 'Eval:\|New best\|Collapse' \"\${d}selfplay.log\" 2>/dev/null | tail -3; echo; done"
