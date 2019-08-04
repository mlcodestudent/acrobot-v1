from typing import Optional, List

from gym.spaces import Box, Discrete
from keras import Model, Sequential
from keras.layers import Dense


def test_models(observation_space: Box, action_space: Discrete) -> [Sequential]:
    models: List[Sequential] = []

    for i in range(2, 32, 2):
        model = _create_one_layer_model(input_shape=observation_space.shape, actions_count=action_space.n, units=i)
        models.append(model)

    for i in range(2, 32, 2):
        for j in range(2, 32, 2):
            model = _create_two_layers__model(input_shape=observation_space.shape, actions_count=action_space.n,
                                              units1=i, units2=j)
            models.append(model)

    return models


def _create_one_layer_model(input_shape: Optional[tuple], actions_count: int, units: int) -> Sequential:
    model = Sequential()
    model.name = "one-hidden-layer-" + str(units)
    model.add(Dense(units=units, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=actions_count, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model


def _create_two_layers__model(input_shape: Optional[tuple], actions_count: int, units1: int, units2: int) -> Sequential:
    model = Sequential()
    model.name = "two-hidden-layer-" + str(units1) + "-" + str(units2)
    model.add(Dense(units=units1, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=units2, activation='relu'))
    model.add(Dense(units=actions_count, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model
