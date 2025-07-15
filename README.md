# RL_Satellite_management

This project implements a reinforcement learning (RL) agent capable of autonomously controlling a satellite to explore and observe an asteroid through multiple predefined orbital paths. The agent must maximize data collection while managing limited onboard resources, such as battery and data buffer, and minimizing delta-v consumption.

## Project Overview

- **Environment**: Custom Gym environment with 5 orbits, each defined by Relative Orbital Elements (ROEs) and discretized into 5 anomaly points.
- **Agent**: Deep Q-Network (DQN) trained using `stable-baselines3` with PyTorch backend.
- **Objective**: Fully explore all orbits while minimizing the number of orbital transfers and energy consumption.
- **Constraints**:
  - Limited battery capacity and data buffer
  - Energy loss at each timestep and during transfers
  - Reward shaping to penalize inefficient behavior and encourage complete exploration

## RL Setup

- **Algorithm**: DQN (`stable-baselines3`)
- **Architecture**: MLP with hidden layers [256, 256]
- **Replay Buffer**: 100,000 transitions
- **Batch Size**: 256
- **Learning Rate**: 1e-4
- **Exploration**: ε-greedy with decay from 1.0 to 0.05 over 60% of training
- **Target Update Interval**: 30,000 steps
- **Training Length**: 4 million steps (~3 hours per run)

## Results

- The agent converges toward stable policies with consistent episode lengths around 160 steps.
- Successful missions correspond to complete exploration of all orbits with optimized delta-v usage (~1.04 m/s total).
- Learned behavior includes energy-aware battery recharge and buffer management, with minimal unnecessary transfers.

