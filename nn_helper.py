import gymnasium as gym
import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import callbacks

# from rl.memory import EpisodeParameterMemory


# def flatten_observation(observation: gym.spaces.Dict) -> np.ndarray:
#     flattened_observation = []
#     for key, value in observation.items():
#         if isinstance(value, dict):
#             for subkey, subvalue in value.items():
#                 flattened_observation.extend(subvalue.reshape(-1))
#         else:
#             flattened_observation.extend(value.reshape(-1))
#     return np.array(flattened_observation)


class DQNAgent:
    def __init__(self, nb_episodes, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=nb_episodes)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 1e-1
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
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


# def create_model(env: gym.Env):
#     model = Sequential()
#     model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#     model.add(Dense(16))
#     model.add(Activation("relu"))
#     model.add(Dense(16))
#     model.add(Activation("relu"))
#     model.add(Dense(16))
#     model.add(Activation("relu"))
#     model.add(Dense(env.action_space.n))
#     model.add(Activation("linear"))

#     print(model.summary())

#     # model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

#     # memory = EpisodeParameterMemory(limit=1000, window_length=1)

#     # cem = CEMAgent(
#     #     model=model,
#     #     nb_actions=output_dim,
#     #     memory=memory,
#     #     batch_size=32,
#     #     nb_steps_warmup=2000,
#     #     train_interval=50,
#     #     elite_frac=0.05,
#     # )

#     # cem.compile()

#     memory = SequentialMemory(limit=50000, window_length=1)
#     policy = BoltzmannQPolicy()
#     dqn = DQNAgent(
#         model=model,
#         nb_actions=env.action_space.n,
#         memory=memory,
#         nb_steps_warmup=10,
#         target_model_update=1e-2,
#         policy=policy,
#     )
#     dqn.compile(adam_v2.Adam(learning_rate=1e-1), metrics=["mae"])

#     return dqn


# Thank you ChatGPT.
def calculate_reward(
    player_position, player_velocity, coins, max_distance, coins_collected
):
    # Calculate distance between player and each coin
    distances = np.linalg.norm(coins - player_position)

    # Find the closest coin
    closest_coin_distance = np.min(distances)

    # Reward based on proximity to closest coin
    proximity_reward = 1.0 - closest_coin_distance / max_distance * coins_collected

    # Penalty based on player velocity (optional)
    velocity_penalty = np.linalg.norm(player_velocity) * 0.01

    # Total reward
    reward = proximity_reward - velocity_penalty

    # print(reward)

    return reward


def train_model(
    model: DQNAgent, env: gym.Env, epochs=100, batch_size=32, sample_weight=[]
):
    # earlystopping_train = callbacks.EarlyStopping(
    #     monitor="loss", mode="min", patience=2, restore_best_weights=True
    # )
    earlystopping_test = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        env,
        nb_steps=50000,
        visualize=True,
        verbose=2,
        callbacks=[earlystopping_test],
    )
    # model.fit(None, nb_steps=100000, visualize=False)


def evaluate_model(model: DQNAgent, env: gym.Env, nb_episodes=5):
    model.test(env, nb_episodes=nb_episodes, visualize=True)


# Finally, evaluate our algorithm for 5 episodes.


# def train_model(
#     model: Sequential, x, y, x_test, y_test, epochs=100, batch_size=32, sample_weight=[]
# ):
#     # earlystopping_train = callbacks.EarlyStopping(
#     #     monitor="loss", mode="min", patience=2, restore_best_weights=True
#     # )
#     earlystopping_test = callbacks.EarlyStopping(
#         monitor="val_loss",
#         mode="min",
#         patience=5,
#         restore_best_weights=True,
#     )
#     model.fit(
#         x,
#         y,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=(x_test, y_test),
#         callbacks=[earlystopping_test],
#         workers=-1,
#         sample_weight=sample_weight,
#     )
#     # model.fit(None, nb_steps=100000, visualize=False)


# def run(
#     env: gym.Env,
#     model: Sequential = None,
# ):
#     train_model(model, env)

#     scores = evaluate_model(model, data_test[0], data_test[1])
#     print(tb([["Loss", scores[0]], ["Accuracy", scores[1]]]))
#     return model
