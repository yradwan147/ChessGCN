#!/bin/bash
# ============================================================
# Opening Suite Sweep: find optimal game start ratios
# ============================================================
#
# All runs: 192h/4blk/1head, Gumbel 32, frozen, 52% gate,
#           buffer window 10, eval every 5, 40 iterations
#
# Variables: opening ratio, FEN pool ratio, games per iter
# Control: no openings (reproduces prior config)
#
# Usage: bash slurm/submit_opening_sweep.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32
BASE="--freeze-value --iterations 40 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5"

echo "============================================================"
echo "ChessGCN Opening Suite Sweep (10 runs, 40 iters each)"
echo "============================================================"
echo ""

# ── Controls (no opening book) ──
echo "── Controls ──"

# o1: No openings, old FEN ratio (reproduces c3_window10 baseline)
echo "  o1: No book, 50% FEN pool (baseline)"
sbatch -J o1_no_book slurm/run_experiment.sh o1_no_book $H $HEADS $BLK $SIMS 3 \
    $BASE --fen-pool-ratio 0.5

# o2: No openings, reduced FEN pool
echo "  o2: No book, 10% FEN pool"
sbatch -J o2_low_fen slurm/run_experiment.sh o2_low_fen $H $HEADS $BLK $SIMS 3 \
    $BASE --fen-pool-ratio 0.1

COUNT=$((COUNT + 2))

# ── Opening ratio sweep ──
echo ""
echo "── Opening ratio sweep (FEN pool fixed at 10%) ──"

# o3: 50% book, 10% FEN, 40% standard
echo "  o3: 50% book, 10% FEN, 40% standard"
sbatch -J o3_book50 slurm/run_experiment.sh o3_book50 $H $HEADS $BLK $SIMS 3 \
    $BASE --openings-file openings.txt --opening-ratio 0.5 --fen-pool-ratio 0.1

# o4: 70% book, 10% FEN, 20% standard (plan default)
echo "  o4: 70% book, 10% FEN, 20% standard"
sbatch -J o4_book70 slurm/run_experiment.sh o4_book70 $H $HEADS $BLK $SIMS 3 \
    $BASE --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.1

# o5: 90% book, 0% FEN, 10% standard
echo "  o5: 90% book, 0% FEN, 10% standard"
sbatch -J o5_book90 slurm/run_experiment.sh o5_book90 $H $HEADS $BLK $SIMS 3 \
    $BASE --openings-file openings.txt --opening-ratio 0.9 --fen-pool-ratio 0.0

COUNT=$((COUNT + 3))

# ── FEN pool ratio sweep (opening ratio fixed at 70%) ──
echo ""
echo "── FEN pool sweep (book fixed at 70%) ──"

# o6: 70% book, 0% FEN, 30% standard
echo "  o6: 70% book, 0% FEN, 30% standard"
sbatch -J o6_nofen slurm/run_experiment.sh o6_nofen $H $HEADS $BLK $SIMS 3 \
    $BASE --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.0

# o7: 70% book, 20% FEN, 10% standard
echo "  o7: 70% book, 20% FEN, 10% standard"
sbatch -J o7_fen20 slurm/run_experiment.sh o7_fen20 $H $HEADS $BLK $SIMS 3 \
    $BASE --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.2

COUNT=$((COUNT + 2))

# ── More games with openings ──
echo ""
echo "── Data volume with openings ──"

# o8: 70% book, 5 games/iter (more data per iter)
echo "  o8: 70% book, 5 games/iter"
sbatch -J o8_5games slurm/run_experiment.sh o8_5games $H $HEADS $BLK $SIMS 5 \
    $BASE --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.1

# o9: 50% book, 5 games/iter
echo "  o9: 50% book, 5 games/iter"
sbatch -J o9_5g_book50 slurm/run_experiment.sh o9_5g_book50 $H $HEADS $BLK $SIMS 5 \
    $BASE --openings-file openings.txt --opening-ratio 0.5 --fen-pool-ratio 0.1

# o10: 70% book, 5 games, 0% FEN
echo "  o10: 70% book, 5 games, 0% FEN"
sbatch -J o10_5g_nofen slurm/run_experiment.sh o10_5g_nofen $H $HEADS $BLK $SIMS 5 \
    $BASE --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.0

COUNT=$((COUNT + 3))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs (40 iters each, ~4-6h per run)"
echo "============================================================"
echo ""
echo "  CONTROLS:"
echo "    o1_no_book    — 50% FEN, no book (baseline)"
echo "    o2_low_fen    — 10% FEN, no book"
echo ""
echo "  OPENING RATIO (FEN=10%):"
echo "    o3_book50     — 50% book, 10% FEN, 40% standard"
echo "    o4_book70     — 70% book, 10% FEN, 20% standard"
echo "    o5_book90     — 90% book,  0% FEN, 10% standard"
echo ""
echo "  FEN POOL RATIO (book=70%):"
echo "    o6_nofen      — 70% book,  0% FEN, 30% standard"
echo "    o7_fen20      — 70% book, 20% FEN, 10% standard"
echo ""
echo "  MORE GAMES:"
echo "    o8_5games     — 70% book, 10% FEN, 5 games/iter"
echo "    o9_5g_book50  — 50% book, 10% FEN, 5 games/iter"
echo "    o10_5g_nofen  — 70% book,  0% FEN, 5 games/iter"
echo ""
echo "Key comparisons:"
echo "  o1 vs o2:           Does reducing FEN pool alone help?"
echo "  o1 vs o3/o4/o5:     Does opening suite help at all?"
echo "  o3 vs o4 vs o5:     Optimal book ratio (50/70/90%)"
echo "  o4 vs o6 vs o7:     Optimal FEN ratio (0/10/20%)"
echo "  o4 vs o8:           More games with book (3 vs 5)"
echo "  o3 vs o9:           50% book at 3g vs 5g"
echo "============================================================"
echo ""
echo "Quick status:"
echo "  for d in results/o*/; do printf '%-20s ' \"\$d\"; grep 'Challenger' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o '[0-9.]*%' || echo 'pending'; done"
