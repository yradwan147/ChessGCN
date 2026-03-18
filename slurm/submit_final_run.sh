#!/bin/bash
# ============================================================
# Final Run: best config, push to the limit
# ============================================================
#
# Settled configuration from 22 experiments:
#   Architecture: 192h, 4blk, 1head (~950K params)
#   MCTS: Gumbel, 32 sims
#   Training: frozen value, 52% gate, eval every 5, 200 steps
#   Anti-collapse: buffer window 10
#
# This run: 300 iterations, 48h time limit.
# c3_window10 was still improving at iter 60 (79.2% eval).
# Let's see how far it goes.
#
# Usage: bash slurm/submit_final_run.sh

set -e
mkdir -p slurm/slurm_logs results

echo "============================================================"
echo "ChessGCN Final Run — 300 iterations, 48h limit"
echo "============================================================"
echo ""

sbatch --time=48:00:00 -J final_300 slurm/run_experiment.sh final_300 192 1 4 32 3 \
    --freeze-value \
    --iterations 300 \
    --eval-interval 5 \
    --gate-threshold 0.52 \
    --buffer-window 10 \
    --save-interval 10

echo ""
echo "Submitted: final_300"
echo "  192h 4blk 1head, Gumbel 32 sims, frozen value"
echo "  300 iters, 3 games/iter, 200 train steps"
echo "  52% gate, eval every 5, buffer window 10"
echo "  Time limit: 48 hours"
echo ""
echo "Estimated: ~30-35h"
echo ""
echo "Monitor:"
echo "  tail -f slurm/slurm_logs/final_300_*.out"
echo "  grep 'Eval:\|New best\|policy_loss' results/final_300/selfplay.log | tail -20"
