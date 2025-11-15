import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from simulation.multi_upc_env import MultiUPCEnv
from rl.double_dqn_agent import DoubleDQNAgent


def clean_state(state_dict: dict) -> np.ndarray:
    arr = np.array(list(state_dict.values()), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


ACTION_SPACE = [-0.10, -0.05, 0.0, 0.05, 0.10]


def main():
    features_path = "panel_augmented2.parquet"
    model_path = "market_simulator_model2.joblib"

    features_df = pd.read_parquet(features_path)
    demand_model = joblib.load(model_path)

    upc_list = features_df["upc"].unique()[:10]

    env = MultiUPCEnv(
        demand_model=demand_model,
        features_df=features_df,
        upc_list=upc_list,
        noise_std=0.02,
    )

    raw_state = env.reset()
    state_dim = len(raw_state)

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=len(ACTION_SPACE),
        lr=1e-4,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.999,
        batch_size=64,
    )

    EPISODES = 1000
    TARGET_UPDATE_EVERY = 20

    episode_rewards = []
    episode_profits = []
    losses = []

    for ep in range(1, EPISODES + 1):
        raw_state = env.reset()
        state = clean_state(raw_state)
        done = False
        total_reward = 0.0
        total_profit = 0.0

        while not done:
            action_idx = agent.select_action(state)
            next_raw_state, reward, done, info = env.step(action_idx, ACTION_SPACE)

            if next_raw_state is None:
                next_state = np.zeros_like(state, dtype=np.float32)
            else:
                next_state = clean_state(next_raw_state)

            agent.store(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            total_profit += info["profit"]

        if ep % TARGET_UPDATE_EVERY == 0:
            agent.update_target()

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_profits.append(total_profit)

        if ep % 10 == 0:
            print(
                f"Episode {ep}/{EPISODES} "
                f"Reward: {total_reward:.2f} "
                f"Profit: {total_profit:.2f} "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title("DQN Normalized Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(episode_profits)
    plt.title("DQN Actual Profit per Episode ($)")
    plt.xlabel("Episode")
    plt.ylabel("Profit ($)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    torch.save(agent.policy_net.state_dict(), "double_dqn_multi_upc.pt")

    agent.epsilon = 0.0
    N_TEST_EPISODES = 100
    test_rewards = []
    test_profits = []

    for _ in range(N_TEST_EPISODES):
        raw_state = env.reset()
        state = clean_state(raw_state)
        done = False
        total_reward = 0.0
        total_profit = 0.0

        while not done:
            action_idx = agent.select_action(state)
            next_raw_state, reward, done, info = env.step(action_idx, ACTION_SPACE)

            if next_raw_state is None:
                next_state = np.zeros_like(state, dtype=np.float32)
            else:
                next_state = clean_state(next_raw_state)

            state = next_state
            total_reward += reward
            total_profit += info["profit"]

        test_rewards.append(total_reward)
        test_profits.append(total_profit)

    print(f"\nAverage Test Reward: {np.mean(test_rewards):.2f}")
    print(f"Average Test Profit: ${np.mean(test_profits):.2f}")


if __name__ == "__main__":
    main()
