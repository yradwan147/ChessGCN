#!/bin/bash
# ============================================================
# Phase 2 Research Sweep: Architecture Changes
# ============================================================
# 6 runs, 40 iters each, testing: attention pooling, moves-left head,
# score distribution head (128 bins)
#
# Usage: bash slurm/submit_research_p2.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32
# Base flags (old features to isolate arch changes)
BASE="--freeze-value --iterations 40 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5 --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.1 --no-self-edges --no-check-feature --wdl-k 111"

echo "============================================================"
echo "Phase 2 Research Sweep (6 runs, 40 iters)"
echo "============================================================"
echo ""

# a1: Baseline (no arch changes, same as old config)
echo "  a1: Baseline (no arch changes)"
sbatch --time=36:00:00 -J a1_baseline slurm/run_experiment.sh a1_baseline $H $HEADS $BLK $SIMS 3 $BASE

# a2: Attention pooling only
echo "  a2: Attention pooling"
sbatch --time=36:00:00 -J a2_attnpool slurm/run_experiment.sh a2_attnpool $H $HEADS $BLK $SIMS 3 $BASE --attn-pool

# a3: Moves-left head only
echo "  a3: Moves-left head"
sbatch --time=36:00:00 -J a3_movesleft slurm/run_experiment.sh a3_movesleft $H $HEADS $BLK $SIMS 3 $BASE --moves-left-head

# a4: Score distribution head (128 bins)
echo "  a4: Score distribution (128 bins)"
sbatch --time=36:00:00 -J a4_scoredist slurm/run_experiment.sh a4_scoredist $H $HEADS $BLK $SIMS 3 $BASE --score-dist-bins 128

# a5: Attention pool + moves-left (no score dist)
echo "  a5: Attn pool + moves-left"
sbatch --time=36:00:00 -J a5_arch_combo slurm/run_experiment.sh a5_arch_combo $H $HEADS $BLK $SIMS 3 $BASE --attn-pool --moves-left-head

# a6: Everything (all 3 arch changes)
echo "  a6: All Phase 2 changes"
sbatch --time=36:00:00 -J a6_all_p2 slurm/run_experiment.sh a6_all_p2 $H $HEADS $BLK $SIMS 3 $BASE --attn-pool --moves-left-head --score-dist-bins 128

COUNT=$((COUNT + 6))

echo ""
echo "============================================================"
echo "Submitted $COUNT Phase 2 jobs"
echo "============================================================"
echo ""
echo "  a1 baseline     — no arch changes (control)"
echo "  a2 attnpool     — GlobalAttention pooling (AlphaGateau)"
echo "  a3 movesleft    — moves-left aux head (Czech et al.)"
echo "  a4 scoredist    — 128-bin score distribution (DeepMind 2024)"
echo "  a5 arch_combo   — attn pool + moves-left"
echo "  a6 all_p2       — all 3 arch changes"
echo ""
echo "Comparisons:"
echo "  a1 vs a2: attention pooling impact"
echo "  a1 vs a3: moves-left head impact"
echo "  a1 vs a4: score distribution impact"
echo "  a5 vs a2+a3: do they combine well?"
echo "  a6: everything together"
echo ""
echo "Quick status:"
echo "  for d in results/a*/; do printf '%-20s ' \"\$d\"; grep 'Challenger' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o '[0-9.]*%' || echo 'pending'; done"
