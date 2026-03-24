"""
Test Script - Verify Environment and Agent Initialization

This script tests that the environment and agent are properly configured
and can run basic interactions.

Usage:
    python test_setup.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.meis_env import MEISEnv
from agents.a3c_agent import A3CAgent
from baselines.s_s_policy import sSPolicy


def test_environment():
    """Test environment creation and basic operations"""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT")
    print("="*60)
    
    # Create environment
    env = MEISEnv()
    print("✓ Environment created successfully")
    
    # Check spaces
    print(f"✓ State space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.n}")
    
    # Reset environment
    state = env.reset()
    print(f"✓ Environment reset, state shape: {state.shape}")
    print(f"  Initial state: {state}")
    
    # Take random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"✓ Episode completed after {step + 1} steps")
            break
    
    print(f"✓ 10 random steps executed successfully")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Service level: {info['service_level']:.2%}")
    
    return True


def test_agent():
    """Test agent creation and action selection"""
    print("\n" + "="*60)
    print("TESTING A3C AGENT")
    print("="*60)
    
    # Create environment
    env = MEISEnv()
    
    # Create agent
    agent = A3CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    print("✓ Agent created successfully")
    
    # Test action selection
    state = env.reset()
    action, log_prob, value = agent.select_action(state)
    print(f"✓ Action selection works")
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value estimate: {value:.2f}")
    
    # Test update (dummy data)
    states = [state for _ in range(10)]
    actions = [env.action_space.sample() for _ in range(10)]
    advantages = np.random.randn(10)
    returns = np.random.randn(10)
    
    metrics = agent.update(states, actions, advantages, returns)
    print(f"✓ Agent update works")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Actor loss: {metrics['actor_loss']:.4f}")
    print(f"  Critic loss: {metrics['critic_loss']:.4f}")
    
    return True


def test_baseline():
    """Test baseline policy"""
    print("\n" + "="*60)
    print("TESTING BASELINE POLICY")
    print("="*60)
    
    # Create environment and policy
    env = MEISEnv()
    policy = sSPolicy()
    print("✓ Baseline policy created successfully")
    
    # Test action selection
    state = env.reset()
    action = policy.select_action(state, env.reorder_quantities)
    print(f"✓ Baseline action selection works")
    print(f"  Action: {action}")
    
    # Run a short episode
    total_reward = 0
    for step in range(50):
        action = policy.select_action(state, env.reorder_quantities)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"✓ Baseline policy episode completed")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps: {step + 1}")
    
    return True


def test_integration():
    """Test full training loop (short version)"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    env = MEISEnv()
    agent = A3CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cpu'
    )
    
    # Short training loop
    n_episodes = 5
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        states, actions, rewards, values, dones = [], [], [], [], []
        
        while not done:
            action, _, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
        
        # Update agent
        if len(states) > 0:
            _, _, next_value = agent.select_action(state)
            advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
            _ = agent.update(states, actions, advantages, returns)
        
        print(f"  Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}")
    
    print("✓ Integration test passed")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#  MEIS RL - SETUP VERIFICATION TEST SUITE" + " "*17 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Environment", test_environment),
        ("A3C Agent", test_agent),
        ("Baseline Policy", test_baseline),
        ("Integration", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED with error:")
            print(f"  {str(e)}")
            results.append((test_name, "FAILED"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
    
    all_passed = all(result == "PASSED" for _, result in results)
    
    if all_passed:
        print("\n" + "🎉 All tests PASSED! Environment is ready for training. 🎉")
        return 0
    else:
        print("\n" + "⚠️  Some tests FAILED. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
