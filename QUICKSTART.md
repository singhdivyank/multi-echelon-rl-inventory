# Quick Start Guide

This guide will help you get started with training and evaluating the MEIS RL system in 5 minutes.

## Prerequisites

Ensure you have Python 3.7+ installed with pip.

## Step 1: Installation (1 minute)

```bash
# Navigate to project directory
cd < directory_name >

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify Setup (1 minute)

Run the test script to ensure everything is working:

```bash
python test_setup.py
```

You should see:
```
🎉 All tests PASSED! Environment is ready for training. 🎉
```

## Step 3: Quick Training Run (2 minutes)

Train the A3C agent for a short demo (50 episodes):

```bash
python main.py --seed 42 --save-dir ./quick_test
```

This will:
- Create and train an A3C agent
- Tune the baseline (s,S) policy
- Evaluate both policies
- Generate comparison plots

Expected output:
```
Starting training for 500 episodes...
Training: 100%|██████████| 500/500 [05:30<00:00,  1.51it/s]

Evaluation results:
  A3C Cost: 125432.45 ± 5234.12
  Baseline Cost: 138765.43 ± 6123.45
  Cost Improvement: 9.61%
```

## Step 4: View Results (1 minute)

Check the generated files:

```bash
ls -R quick_test/

# You should see:
# quick_test/
# ├── checkpoints/
# │   ├── checkpoint_final.pt
# │   └── training_history.json
# ├── plots/
# │   ├── training_curves.png
# │   ├── comparison.png
# │   ├── policy_heatmap.png
# │   └── cost_per_period.png
# └── logs/
#     ├── config.json
#     ├── evaluation_results.json
#     └── baseline_params.json
```

View the plots to see:
- Training progress
- RL vs Baseline comparison
- Policy analysis

## Step 5: Multi-Seed Evaluation (Optional)

Evaluate variance across multiple seeds:

```bash
python run_multi_seed.py \
    --seeds 42 123 456 \
    --n-eval 50 \
    --checkpoint-dir ./quick_test \
    --save-dir ./quick_test
```

## Common Commands

### Train with custom config
```bash
python main.py --config configs/custom.yaml --seed 42
```

### Evaluate only (skip training)
```bash
python main.py --eval-only --save-dir ./results
```

### Full training run (500 episodes)
```bash
python main.py --seed 42 --save-dir ./results/full_run
```

### Interactive plots
```bash
python main.py --seed 42 --show-plots
```

## What's Next?

1. **Customize the environment**: Edit `envs/meis_env.py`
2. **Tune hyperparameters**: Modify `configs/default.yaml`
3. **Run experiments**: Use different seeds and configurations
4. **Analyze results**: Check the generated plots and JSON files

## Troubleshooting

**Issue: ModuleNotFoundError**
```bash
# Make sure you're in the project directory
cd < directory_name >
# And try running with python -m
python -m main --seed 42
```

**Issue: CUDA out of memory**
```bash
# Force CPU usage
python main.py --seed 42 --config configs/default.yaml
# (Edit config to set device: 'cpu')
```

**Issue: Training too slow**
```bash
# Reduce number of episodes in config
# Edit configs/default.yaml:
#   training:
#     n_episodes: 100  # Instead of 500
```

## Understanding the Output

### Training Progress
```
Training:  20%|██  | 100/500 [02:15<09:00,  1.35s/it, 
           reward=-125432, avg_reward_100=-128456, service_level=92.34%]
```
- `reward`: Current episode reward (negative cost)
- `avg_reward_100`: Moving average over last 100 episodes
- `service_level`: Percentage of demand met immediately

### Evaluation Results
```
A3C Agent:
  Mean Cost: 125432.45 ± 5234.12
  Mean Service Level: 94.23% ± 0.78%

Baseline:
  Mean Cost: 138765.43 ± 6123.45
  Mean Service Level: 91.45% ± 1.12%

Improvement:
  Cost: 9.61%  ← Lower is better!
  Service Level: 2.78%  ← Higher is better!
```

### Interpreting Plots

1. **training_curves.png**: Shows learning progress
   - Reward should increase (less negative) over time
   - Service level should stabilize around 90-95%

2. **comparison.png**: Bar charts comparing RL vs Baseline
   - RL should have lower cost bars
   - RL should have higher service level bars

3. **policy_heatmap.png**: Action distribution
   - Shows which reorder actions are preferred
   - Different from baseline indicates learned behavior

4. **cost_per_period.png**: Cost trajectory over time
   - Both policies should show similar patterns
   - RL should have lower average cost (green line below red)

## Success Criteria

Your training is successful if:
- ✓ Training converges (reward stabilizes)
- ✓ Service level > 90%
- ✓ RL cost < Baseline cost (or within 5%)
- ✓ No NaN or infinity values in metrics

## Getting Help

- Check the main README.md for detailed documentation
- Review the code comments for implementation details
- Open an issue on GitHub for bugs or questions

Happy training! 🚀