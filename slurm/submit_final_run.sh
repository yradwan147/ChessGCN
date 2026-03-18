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
# This run: 150 iterations to see how far the model can go.
# c3_window10 was still improving at iter 60 (79.2% eval).
#
# Usage: bash slurm/submit_final_run.sh

set -e
mkdir -p slurm/slurm_logs results

echo "============================================================"
echo "ChessGCN Final Run — 150 iterations, best config"
echo "============================================================"
echo ""

sbatch -J final_150 slurm/run_experiment.sh final_150 192 1 4 32 3 \
    --freeze-value \
    --iterations 150 \
    --eval-interval 5 \
    --gate-threshold 0.52 \
    --buffer-window 10 \
    --save-interval 5

echo ""
echo "Submitted: final_150"
echo "  192h 4blk 1head, Gumbel 32 sims, frozen value"
echo "  150 iters, 3 games/iter, 200 train steps"
echo "  52% gate, eval every 5, buffer window 10"
echo ""
echo "Estimated: ~15-18h"
echo ""
echo "Monitor:"
echo "  tail -f slurm/slurm_logs/final_150_*.out"
echo "  grep 'Eval:\|New best\|policy_loss' results/final_150/selfplay.log | tail -20"
