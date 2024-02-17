"""
Reinforcement Unsupervised Learning Neural Network Model v2
Author: minhcrafters
With some code from StackOverflow :)
"""

import tensorflow as tf
import nn_helper as nn
import numpy as np

from keras.callbacks import TensorBoard
from tabulate import tabulate as tb
from datetime import datetime
from coin_collector_env import CoinCollectorEnv

log_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

if __name__ == "__main__":
    # Import the CoinCollectorEnv class and create an environment object for a human player
    env = CoinCollectorEnv("human")

    # Set up hyperparameters for training the PPO agent
    N = 50  # Number of steps before learning
    batch_size = 128  # Size of batches for training
    n_epochs = 24  # Number of epochs for training
    alpha = 0.0003  # Learning rate for the agent

    # Create a PPOAgent object with the specified parameters
    agent = nn.PPOAgent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    # Set the number of games to play
    n_games = 900

    # Initialize variables to track the best score, score history, learning iterations, average score, and total steps
    best_score = 0
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    # Loop through the specified number of games
    for i in range(n_games):
        # Reset the environment and get the initial observation and info
        observation, info = env.reset()
        done = False
        score = 0

        # Play the game until it's done
        while not done:
            # Choose an action using the PPO agent and take a step in the environment
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action, i)
            n_steps += 1
            score += reward

            # Store the transition in the agent's memory
            agent.store_transition(observation, action, prob, val, reward, done)

            # Check if it's time to learn
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            # Update the current observation
            observation = observation_

        # Update the score history and calculate the average score
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # Check if the average score is better than the best score seen so far
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # Log the reward summary using TensorBoard
        tf.summary.scalar("Reward summary", data=avg_score, step=i)

        # Print the episode details
        print(
            tb(
                [
                    ["Episode", i],
                    ["Score", score],
                    ["Avg. score", avg_score],
                    ["Time steps", n_steps],
                    ["Learning steps", learn_iters],
                ]
            )
        )

        # Render the environment
        env.render(i)
    # filename = "ppo.png"
    # x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
