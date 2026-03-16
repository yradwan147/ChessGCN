#!/bin/bash
# ============================================================
# Architecture + MCTS Sweep: 6 parallel experiments
# ============================================================
#
# Settled decisions (all runs):
#   - Gumbel MCTS (10x faster than PUCT)
#   - Value head frozen (unfreezing caused drift)
#   - Smooth temp annealing, 12 eval games, 55% gate
#   - 30 iterations, buffer 300K, 200 train steps
#
# Variables under test:
#   - Architecture: hidden dim (128 vs 192), blocks (4 vs 5), heads (1 vs 4)
#   - MCTS: sims (32 vs 64), games/iter (3 vs 5)
#
# Usage: bash slurm/submit_arch_sweep.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0

echo "============================================================"
echo "ChessGCN Architecture + MCTS Sweep (6 runs)"
echo "============================================================"
echo ""
echo "All runs use: Gumbel MCTS, frozen value head, 30 iters"
echo ""

# ---------------------------------------------------------------
# Args: RUN_TAG HIDDEN HEADS BLOCKS SIMS GAMES
# ---------------------------------------------------------------

# Run 1: Current arch baseline (128h, 4blk, 4head, 32 sims)
# Purpose: clean reference with frozen value head + Gumbel
echo "Run 1: Baseline — 128h, 4blk, 4head, 32 sims, 3 games"
sbatch -J r1_baseline slurm/run_experiment.sh r1_baseline 128 4 4 32 3
COUNT=$((COUNT + 1))

# Run 2: Single-head attention (128h, 4blk, 1head, 32 sims)
# Purpose: test AlphaGateau's finding that 1 head > 4 heads
echo "Run 2: Single head — 128h, 4blk, 1head, 32 sims, 3 games"
sbatch -J r2_1head slurm/run_experiment.sh r2_1head 128 1 4 32 3
COUNT=$((COUNT + 1))

# Run 3: Wider model (192h, 4blk, 1head, 32 sims)
# Purpose: test width scaling (~950K params, 2.2x current)
echo "Run 3: Wider — 192h, 4blk, 1head, 32 sims, 3 games"
sbatch -J r3_wide slurm/run_experiment.sh r3_wide 192 1 4 32 3
COUNT=$((COUNT + 1))

# Run 4: Deeper model (128h, 5blk, 1head, 32 sims)
# Purpose: test depth scaling (match AlphaGateau's 5 blocks)
echo "Run 4: Deeper — 128h, 5blk, 1head, 32 sims, 3 games"
sbatch -J r4_deep slurm/run_experiment.sh r4_deep 128 1 5 32 3
COUNT=$((COUNT + 1))

# Run 5: AlphaGateau scale (192h, 5blk, 1head, 32 sims)
# Purpose: closest to proven AlphaGateau config (~1.4M params)
echo "Run 5: AlphaGateau scale — 192h, 5blk, 1head, 32 sims, 3 games"
sbatch -J r5_agscale slurm/run_experiment.sh r5_agscale 192 1 5 32 3
COUNT=$((COUNT + 1))

# Run 6: Big model + more search (192h, 5blk, 1head, 64 sims, 5 games)
# Purpose: does bigger model benefit from more sims + data?
echo "Run 6: Big + search — 192h, 5blk, 1head, 64 sims, 5 games"
sbatch -J r6_big_search slurm/run_experiment.sh r6_big_search 192 1 5 64 5
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs"
echo ""
echo "  Run 1: r1_baseline   128h 4blk 4head 32sim 3g  (baseline)"
echo "  Run 2: r2_1head      128h 4blk 1head 32sim 3g  (head count)"
echo "  Run 3: r3_wide       192h 4blk 1head 32sim 3g  (width)"
echo "  Run 4: r4_deep       128h 5blk 1head 32sim 3g  (depth)"
echo "  Run 5: r5_agscale    192h 5blk 1head 32sim 3g  (AlphaGateau)"
echo "  Run 6: r6_big_search 192h 5blk 1head 64sim 5g  (big+search)"
echo ""
echo "Comparisons:"
echo "  1 vs 2: Do fewer heads help?"
echo "  2 vs 3: Does wider hidden help?"
echo "  2 vs 4: Does deeper help?"
echo "  3 vs 4: Width vs depth at similar param count?"
echo "  4 vs 5: Both width + depth?"
echo "  5 vs 6: More search budget for bigger model?"
echo "============================================================"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/slurm_logs/r1_baseline_*.out"
echo ""
echo "Quick status:"
echo "  grep -h 'PHASE\|Iteration\|Eval:\|SUCCESS\|policy_loss' slurm/slurm_logs/r*_*.out | tail -30"
echo ""
echo "Compare policy loss across runs:"
echo "  for d in results/r*/; do echo \"\$d\"; grep 'policy_loss' \"\${d}selfplay.log\" | tail -1; done"
