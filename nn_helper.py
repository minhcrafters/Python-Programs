"""
Reinforcement Unsupervised Learning Neural Network Model v2
Author: minhcrafters
With some code from StackOverflow :)
"""

import keras
import random
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from keras.layers import Dense, Flatten
from keras.models import Sequential
from collections import deque
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, nb_episodes, state_size, action_size):
        """
        Initialize the DQN agent with the given number of episodes, state size, and action size.
        """
        # Initialize the state and action sizes
        self.state_size = state_size
        self.action_size = action_size
        # Create a memory buffer with a maximum length of nb_episodes
        self.memory = deque(maxlen=nb_episodes)
        # Set the discount factor for future rewards
        self.gamma = 0.9
        # Initialize the state to None
        self.state = None
        # Initialize the exploration rate for the agent
        self.epsilon = 1.0
        # Set the decay rate for the exploration rate
        self.epsilon_decay = 0.99
        # Set the minimum exploration rate
        self.epsilon_min = 0.01
        # Set the learning rate for the neural network
        self.learning_rate = 0.01
        # Use MirroredStrategy for distributed training
        self.strategy = tf.distribute.MirroredStrategy()

        # Create the neural network model within the distributed strategy scope
        with self.strategy.scope():
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name):
        self.model.save_weights(name)


class PPOMemory:
    def __init__(self, batch_size):
        """
        Initialize the ReplayBuffer class with the given batch size.

        Parameters:
            batch_size (int): The size of the batch for the replay buffer.

        Returns:
            None
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        """
        Generate batches of states, actions, probabilities, values, rewards, and dones.

        Returns:
            tuple: A tuple containing numpy arrays of states, actions, probabilities, values, rewards, dones, and batches.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")
        self.flatten = Flatten()
        self.fc3 = Dense(n_actions, activation="softmax")

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")
        self.flatten = Flatten()
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        q = self.q(x)

        return q


MAX_STEPS = 10000  # Bonus or penalty for teleportation
MIN_DISTANCE_PENALTY = -10  # Minimum penalty for being far from a coin
MAX_DISTANCE_PENALTY = 10  # Maximum penalty for being far from a coin


def calculate_reward(
    agent_position, coin_position, score, current_gen, previous_agent_position=None
):
    distance = np.linalg.norm(agent_position - coin_position)

    # Calculate distance-based penalty
    distance_penalty = np.interp(
        distance, [-38.5, 800], [MAX_DISTANCE_PENALTY, MIN_DISTANCE_PENALTY]
    )

    # Calculate reward for collecting coin
    coin_reward = score + np.sum(
        np.subtract(agent_position, previous_agent_position)
        if np.sum(agent_position) > np.sum(previous_agent_position)
        else np.subtract(previous_agent_position, agent_position)
    )

    # Calculate total reward
    total_reward = (coin_reward + distance_penalty) * (current_gen / MAX_STEPS)

    return total_reward


class PPOAgent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        chkpt_dir="./models/",
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha), metrics=["accuracy"])
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha), metrics=["accuracy"])
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... Saving models ...")
        self.actor.save(self.chkpt_dir + "actor")
        self.critic.save(self.chkpt_dir + "critic")

    def load_models(self):
        print("... Loading models ...")
        self.actor = keras.models.load_model(self.chkpt_dir + "actor")
        self.critic = keras.models.load_model(self.chkpt_dir + "critic")

    def choose_action(self, observation):
        state = tf.convert_to_tensor(observation)

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value


    def learn(self):
        """
        A method for learning using the Proximal Policy Optimization (PPO) algorithm.
        """

        # Loop through a number of epochs
        for _ in range(self.n_epochs):
            # Generate batches of data from memory
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            # Calculate advantages for each time step
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            # Update the actor and critic networks
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    # Get action probabilities from the actor network
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    # Get critic value prediction from the critic network
                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)

                    # Calculate PPO loss for the actor network
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(
                        prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                    )
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    # Calculate MSE loss for the critic network
                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)

                # Compute and apply gradients for actor and critic networks
                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        # Clear the memory after all epochs
        self.memory.clear_memory()
