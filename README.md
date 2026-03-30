python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train PPO on Divergent environment
python run_divergent.py --mode train

# Evaluate trained model
python run_divergent.py --mode eval --checkpoint results/checkpoints/ppo_divergent_best.pt

# Generate plots and visualizations
python evaluation/visualize.py --experiment divergent --compare-baseline

# Divergent structure (main experiment)
python run_divergent.py

# Generate all visualizations
python evaluation/visualize.py --all