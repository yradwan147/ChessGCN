#!/bin/bash
# ============================================================
# Opening Suite Run: test opening book impact on training
# ============================================================
#
# Best config + opening suite (70% book starts, 20% standard, 10% FEN pool)
# Compare against c3_window10 baseline (no book, 50% FEN pool)
#
# Usage: bash slurm/submit_opening_run.sh

set -e
mkdir -p slurm/slurm_logs results

echo "============================================================"
echo "ChessGCN Opening Suite Experiment"
echo "============================================================"
echo ""

sbatch --time=30:00:00 -J opening_60 slurm/run_experiment.sh opening_60 192 1 4 32 3 \
    --freeze-value \
    --iterations 60 \
    --eval-interval 5 \
    --gate-threshold 0.52 \
    --buffer-window 10 \
    --save-interval 5 \
    --openings-file openings.txt \
    --opening-ratio 0.7 \
    --fen-pool-ratio 0.1

echo ""
echo "Submitted: opening_60"
echo "  192h 4blk 1head, Gumbel 32 sims, frozen value"
echo "  60 iters, 3 games/iter, buffer window 10"
echo "  Opening suite: 70% book starts, 20% standard, 10% FEN pool"
echo ""
echo "Compare with c3_window10 (no book, 50% FEN pool):"
echo "  - Policy loss trajectory"
echo "  - Eval win rate (should still improve)"
echo "  - Game diversity (more varied openings expected)"
echo ""
echo "Monitor:"
echo "  tail -f slurm/slurm_logs/opening_60_*.out"
echo "  grep 'game_type\|Eval:\|policy_loss' results/opening_60/selfplay.log | tail -20"
