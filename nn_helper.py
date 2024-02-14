import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import callbacks
# from rl.agents.cem import CEMAgent
# from rl.memory import EpisodeParameterMemory


def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, activation="relu"))
    model.add(Dense(input_dim, activation="relu"))
    model.add(Dense(input_dim + 2, activation="relu"))
    model.add(Dense(output_dim + 4, activation="relu"))
    model.add(Dense(output_dim, activation="sigmoid"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    # memory = EpisodeParameterMemory(limit=1000, window_length=1)

    # cem = CEMAgent(
    #     model=model,
    #     nb_actions=output_dim,
    #     memory=memory,
    #     batch_size=32,
    #     nb_steps_warmup=2000,
    #     train_interval=50,
    #     elite_frac=0.05,
    # )

    # cem.compile()

    print(model.summary())
    return model


# game_state = [fox_x, fox_y, coin_x, coin_y, relative_dist_x, relative_dist_y]
# 0 - left, 1 - right
# 2 - up, 3 - down
# TODO: implement the data


def preprocess_data(df: pd.DataFrame):
    game_states = df[
        [
            "player_pos_x",
            "player_pos_y",
            "player_vel_x",
            "player_vel_y",
            "player_accel",
            "coin_pos_x",
            "coin_pos_y",
            "rel_dist_x",
            "rel_dist_y",
        ]
    ]
    actions = df[["move_right", "move_left", "move_down", "move_up"]]
    x = np.array(game_states)
    y = np.array(actions)
    return x, y


def train_model(model: Sequential, x, y, x_test, y_test, epochs=100, batch_size=32):
    # earlystopping_train = callbacks.EarlyStopping(
    #     monitor="loss", mode="min", patience=2, restore_best_weights=True
    # )
    earlystopping_test = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )
    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[earlystopping_test],
    )
    # model.fit(None, nb_steps=100000, visualize=False)


def evaluate_model(model: Sequential, x_test, y_test):
    scores = model.evaluate(x_test, y_test)
    return scores


def make_prediction(model: Sequential, game_state: list):
    return model.predict(np.array(game_state))[0]


if __name__ == "__main__":
    dataset_name = "results_25_025_14022024_183513.csv"
    df = pd.read_csv(f"./dataset/{dataset_name}")

    df_train = df.iloc[: df.shape[0] // 2, :]
    df_test = df.iloc[df.shape[0] // 2 :, :]
    data_train = preprocess_data(df_train)
    data_test = preprocess_data(df_test)

    if dataset_name.endswith(".csv"):
        model = create_model(data_train[0].shape[1], data_train[1].shape[1])
    elif dataset_name.endswith(".keras"):
        model = load_model("./model/model_{}.keras".format(dataset_name[8:-4]))
    else:
        raise ValueError("Invalid dataset name")

    train_model(
        model, data_train[0], data_train[1], data_test[0], data_test[1], epochs=650
    )

    print(f"[loss, accuracy]: {evaluate_model(model, data_test[0], data_test[1])}")
    model.save("./model/model_{}.keras".format(dataset_name[8:-4]))
