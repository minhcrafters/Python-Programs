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
    env = CoinCollectorEnv()

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = nn.PPOAgent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(np.expand_dims(observation, axis=0))
            observation_, reward, done, _, info = env.step(action, i)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        tf.summary.scalar("Reward summary", data=avg_score, step=i)
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
        env.render(i)
    # filename = "ppo.png"
    # x = [i + 1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)
