# Multi-Echelon Inventory System with Deep Reinforcement Learning

A complete implementation of Actor-Critic (A3C) reinforcement learning for multi-echelon inventory optimization, based on the research paper analysis. This project implements a divergent 2-layer supply chain with stochastic lead times and demand, comparing RL performance against classical (s,S) inventory policies.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment](#environment)
- [Agent](#agent)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results & Visualization](#results--visualization)
- [Customization](#customization)
- [Project Structure](#project-structure)
- [Research Background](#research-background)

## 🎯 Overview

This project implements:

1. **Multi-Echelon Inventory System (MEIS) Gym Environment**
   - Factory with infinite supply
   - 1 Middle warehouse
   - 2 Leaf warehouses
   - Stochastic demand and lead times
   - Backlog and stockout modeling

2. **A3C (Asynchronous Advantage Actor-Critic) Agent**
   - Fully connected MLP networks (3 layers, 64 neurons)
   - Generalized Advantage Estimation (GAE)
   - Policy entropy regularization
   - Separate actor and critic networks

3. **(s,S) Baseline Policy**
   - Classical reorder point policy
   - Automated parameter tuning via optimization
   - Statistical comparison framework

4. **Comprehensive Evaluation Framework**
   - Multi-seed variance analysis
   - Statistical significance testing
   - Policy visualization and comparison
   - Training curve analysis

## 🏗️ Architecture

### System Design

```
┌─────────────┐
│   Factory   │ (Infinite Supply)
└──────┬──────┘
       │ Orders from Middle
       ↓
┌─────────────┐
│   Middle    │
│  Warehouse  │
└──────┬──────┘
       │ Orders from Leaves
       ├───────────┬───────────┐
       ↓           ↓           ↓
  ┌────────┐  ┌────────┐  Material Flow
  │ Leaf 1 │  │ Leaf 2 │  (Top → Bottom)
  │Warehouse│ │Warehouse│
  └────────┘  └────────┘  Information Flow
      ↑           ↑        (Bottom → Top)
   Demand      Demand
```

### Agent Architecture

**Actor Network** (Policy)
```
Input (12-dim state) → FC(64) → ReLU → FC(64) → ReLU → FC(64) → ReLU → FC(13 actions)
```

**Critic Network** (Value Function)
```
Input (12-dim state) → FC(64) → ReLU → FC(64) → ReLU → FC(64) → ReLU → FC(1 value)
```

## 🌍 Environment

### State Space (12-dimensional)

For each warehouse (Middle, Leaf 1, Leaf 2):
- **Inventory on Hand (IOH)**: Current inventory level
- **Oldest Order Quantity**: Quantity of the oldest pending order
- **Days Since Oldest Order**: Age of the oldest pending order
- **Total Reorder Quantity**: Sum of all open orders

### Action Space (13 discrete actions)

- **Action 0**: No warehouse orders
- **Actions 1-4**: Middle warehouse orders (4 quantity options)
- **Actions 5-8**: Leaf 1 warehouse orders (4 quantity options)
- **Actions 9-12**: Leaf 2 warehouse orders (4 quantity options)

**Reorder Quantities:**
- Middle: [5000, 10000, 15000, 20000]
- Leaf: [3000, 6000, 9000, 12000]

### Reward Function

```
reward = -total_cost
```

Where `total_cost` includes:
1. **Shortage Cost**: Penalty for stockouts (only at leaf warehouses)
2. **Holding Cost**: Cost of maintaining inventory
3. **Reordering Cost**: Fixed and variable ordering costs

### Cost Formulation

```python
# Shortage Cost (leaf warehouses only)
c_shortage = k_shortage × min(0, IOH) × price_per_product

# Holding Cost
c_holding = k_holding × max(0, IOH) × price_per_product

# Reordering Cost
c_reordering = min(c_min_reorder, k_reorder × q_reorder)

# Total Cost
total_cost = Σ(c_shortage + c_holding + c_reordering)
```

### Environment Parameters

| Parameter | Middle Warehouse | Leaf Warehouse |
|-----------|-----------------|----------------|
| Initial IOH | 10,000 | 5,000 |
| Lead Time Mean | 2 days | 2 days |
| Lead Time Std | 1 day | 1 day |
| Demand Mean | - | 3,300 units |
| Demand Std | - | 100 units |
| Price per Product | 50 CHF | 100 CHF |
| Min Reorder Cost | 1,000 CHF | 5,000 CHF |
| Holding Cost Constant | 0.1 | 0.1 |
| Shortage Cost Constant | 0 | 10 |
| Max Backlog Duration | - | 7 days |

## 🤖 Agent

### A3C Algorithm

The Asynchronous Advantage Actor-Critic implementation includes:

1. **Advantage Estimation**: Generalized Advantage Estimation (GAE) with λ=0.95
2. **Policy Gradient**: Actor loss with entropy regularization
3. **Value Function**: Critic loss (MSE with returns)
4. **Optimization**: Adam optimizer with gradient clipping

### Key Hyperparameters

```python
{
    'lr': 1e-4,
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE parameter
    'entropy_coef': 0.01,       # Entropy regularization
    'value_loss_coef': 0.5,     # Critic loss weight
    'max_grad_norm': 0.5,       # Gradient clipping
    'n_layers': 3,              # Network depth
    'hidden_size': 64,          # Neurons per layer
}
```

### Loss Function

```
L_total = L_actor + β_v × L_critic + β_e × L_entropy

where:
  L_actor = -𝔼[log π(a|s) × A(s,a)]
  L_critic = MSE(V(s), returns)
  L_entropy = -𝔼[H(π(·|s))]
```

## 🚀 Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- NumPy, SciPy, Matplotlib, Seaborn
- Gym (OpenAI)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd meis_rl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

<!-- ## ⚡ Quick Start

### Train and Evaluate (Single Run)

```bash
# Train A3C agent and compare with baseline
python main.py --seed 42 --save-dir ./results/run1
```

### Evaluate Across Multiple Seeds

```bash
# First, train a model
python main.py --seed 42 --save-dir ./results

# Then evaluate across multiple seeds
python run_multi_seed.py \
    --seeds 42 123 456 789 1000 \
    --n-eval 100 \
    --checkpoint-dir ./results \
    --save-dir ./results
```

### Evaluate Pre-trained Model Only

```bash
python main.py --eval-only --save-dir ./results
``` -->

## 📚 Training

### Basic Training

### Training with Custom Configuration

Create a YAML config file (`configs/custom.yaml`):

Then run:

```bash
python main.py --config configs/custom.yaml --seed 42 --save-dir ./results
```

### Training Output

During training, you'll see:
- Episode-by-episode progress
- Moving average rewards
- Service level metrics
- Periodic evaluations
- Automatic checkpointing

```
Training:  20%|██        | 100/500 [02:15<09:00,  1.35s/it, reward=-125432.45, avg_reward_100=-128456.23, service_level=92.34%]

Evaluation at episode 100:
  Mean reward: -125432.45 ± 5234.12
  Mean cost: 125432.45 ± 5234.12
  Service level: 92.34%

Checkpoint saved: ./results/checkpoints/checkpoint_ep100.pt
```

## 📊 Evaluation

### Evaluation Metrics

The evaluation framework computes:

1. **Performance Metrics**
   - Mean total cost
   - Cost standard deviation
   - Mean service level
   - Cost breakdown (shortage, holding, reordering)

2. **Statistical Tests**
   - Independent t-test for cost comparison
   - Cohen's d effect size
   - Confidence intervals

3. **Variance Analysis**
   - Cross-seed variance
   - Stability assessment

### Evaluate Pre-trained Model Only

```bash
python main.py --eval-only --save-dir ./results
```

### Example Evaluation Results

```
=============================================================
SUMMARY STATISTICS ACROSS SEEDS
=============================================================

Number of seeds: 5
Seeds: [42, 123, 456, 789, 1000]

--- Cost ---
RL Agent:
  Mean: 125432.45
  Std: 2341.23
  Min: 122456.78
  Max: 128901.34

Baseline:
  Mean: 138765.43
  Std: 3124.56
  Min: 134567.89
  Max: 143210.98

Cost Improvement:
  Mean: 9.61%
  Std: 1.23%
  Min: 7.89%
  Max: 11.45%

--- Service Level ---
RL Agent:
  Mean: 94.23%
  Std: 0.78%

Baseline:
  Mean: 91.45%
  Std: 1.12%

=============================================================
```

## 📈 Results & Visualization

### Automatically Generated Plots

The system generates four main visualizations:

#### 1. Training Curves (`training_curves.png`)
- Episode rewards over time
- Episode costs over time
- Service levels progression
- Training losses (actor, critic, entropy)

#### 2. Comparison Plot (`comparison.png`)
- Bar chart comparing RL vs Baseline costs
- Bar chart comparing service levels
- Error bars showing standard deviation

#### 3. Policy Heatmap (`policy_heatmap.png`)
- Action distribution for RL agent
- Action distribution for baseline
- Shows which actions are preferred

#### 4. Cost Per Period (`cost_per_period.png`)
- Time-series comparison
- Shows cost trajectory over episode
- Shaded regions indicate variance

#### 5. Variance Plots (Multi-seed evaluation)
- `cost_variance.png`: Cost across seeds
- `service_level_variance.png`: Service level across seeds

### Results Directory Structure

```
results/
├── checkpoints/
│   ├── checkpoint_ep100.pt
│   ├── checkpoint_ep200.pt
│   ├── checkpoint_final.pt
│   └── training_history.json
├── plots/
│   ├── training_curves.png
│   ├── comparison.png
│   ├── policy_heatmap.png
│   └── cost_per_period.png
├── logs/
│   ├── config.json
│   ├── evaluation_results.json
│   └── baseline_params.json
└── multi_seed_results/
    ├── seed_results.json
    ├── cost_variance.png
    └── service_level_variance.png
```

## 🔧 Customization

### Modifying the Environment

Edit `envs/meis_env.py` to customize:

```python
# Example: Change demand distribution
def _generate_demand(self):
    for warehouse in ['leaf_1', 'leaf_2']:
        config = self.warehouse_config[warehouse]
        # Change to Poisson distribution
        demand = np.random.poisson(config['demand_mean'])
        # ... rest of the code
```

### Modifying the Agent

Edit `agents/a3c_agent.py` to customize:

```python
# Example: Add recurrent layers
class ActorCriticNetwork(nn.Module):
    def __init__(self, ...):
        # Add LSTM
        self.lstm = nn.LSTM(state_dim, hidden_size)
        # ... rest of the architecture
```

### Custom Baseline Policy

Create a new policy in `baselines/`:

```python
class MyCustomPolicy:
    def select_action(self, state, reorder_quantities):
        # Your custom logic here
        return action
```

## 📂 Project Structure

```
meis_rl/
│
├── envs/
│   └── meis_env.py              # Gym environment implementation
│
├── agents/
│   ├── a3c_agent.py             # A3C agent and network
│   └── trainer.py               # Training loop
│
├── baselines/
│   └── s_s_policy.py            # (s,S) baseline policy
│
├── utils/
│   ├── evaluation.py            # Evaluation utilities
│   └── visualization.py         # Plotting functions
│
├── configs/
│   └── default.yaml             # Default configuration
│
├── experiments/
│   └── (experiment scripts)
│
├── results/
│   └── (training outputs)
│
├── main.py                      # Main training script
├── run_multi_seed.py           # Multi-seed evaluation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 📖 Research Background

This implementation is based on the research paper analysis focusing on:

### Key Concepts

1. **Multi-Echelon Inventory Systems (MEIS)**
   - Hierarchical supply chain structure
   - Information flow (bottom-up) vs material flow (top-down)
   - Divergent topology (one-to-many distribution)

2. **Reinforcement Learning Formulation**
   - State: Inventory positions and pipeline orders
   - Action: Reorder decisions per warehouse
   - Reward: Negative total cost (minimize cost = maximize reward)

3. **Actor-Critic Method**
   - Policy gradient with value function baseline
   - Reduces variance in gradient estimation
   - Entropy regularization prevents premature convergence

4. **Challenges Addressed**
   - Continuous action spaces handled via discretization
   - Stochastic lead times and demand
   - Multi-objective optimization (cost vs service level)
   - Stock-out and backlog modeling

### Comparison with Classical Methods

| Aspect | (s,S) Policy | A3C RL |
|--------|-------------|--------|
| Adaptability | Fixed parameters | Learns from experience |
| Optimality | Local optimum | Potential global optimum |
| Multi-echelon | Separate optimization | Joint optimization |
| Computational Cost | Low (evaluation) | High (training) |
| Interpretability | High | Lower |

## 🎓 Expected Results

Based on the research and implementation:

### Performance Expectations

1. **Cost Reduction**: RL should achieve 5-15% cost reduction vs tuned (s,S) baseline
2. **Service Level**: RL maintains or improves service level (>90%)
3. **Convergence**: Training converges within 300-500 episodes
4. **Variance**: Multi-seed evaluation shows consistent improvement

### Training Characteristics

- **Early Phase (0-100 episodes)**: High variance, exploration
- **Mid Phase (100-300 episodes)**: Convergence begins, variance reduces
- **Late Phase (300-500 episodes)**: Fine-tuning, stable performance

### Baseline Comparison

The (s,S) policy provides a strong baseline because:
- Well-tuned parameters via optimization
- Proven effectiveness in single-echelon systems
- But suboptimal for multi-echelon coordination

## 🐛 Troubleshooting

### Common Issues

**Issue: Training is unstable**
- Solution: Reduce learning rate, increase entropy coefficient
- Check: `agent.learning_rate`, `agent.entropy_coef`

**Issue: Agent always takes same action**
- Solution: Increase exploration (entropy), check action space
- Verify: State normalization, reward scaling

**Issue: Baseline outperforms RL**
- Solution: Increase training episodes, tune hyperparameters
- Check: Network capacity, learning rate schedule

**Issue: High memory usage**
- Solution: Reduce batch size, use gradient accumulation
- Consider: Reducing `n_eval_episodes` during training

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hammler2020multi,
  title={Multi-Echelon Inventory Optimization Using Deep Reinforcement Learning},
  author={Hammler, P. and others},
  journal={Your Conference/Journal},
  year={2020}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gym for the environment framework
- PyTorch team for the deep learning library
- Research paper authors for the theoretical foundation
- Community contributors

---

**Happy Training! 🚀**