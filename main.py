import datetime
import os
import time

from test_models import test_models
from exporation_policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from game import *
from strategies.train_strategy import *
import matplotlib.pyplot as plt

artifacts_dir_path = "../models/"

if not os.path.exists(artifacts_dir_path):
    os.makedirs(artifacts_dir_path)


def train(game: Game, model: Model, epochs: int, games_in_epoch: int) -> [float]:
    exploration_policy = EpsilonGreedyPolicy(max_epochs=epochs)
    strategy = TrainStrategy(model=model, exploration_policy=exploration_policy)
    results = []
    log_file_path = artifacts_dir_path + model.name + ".log"

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    while not exploration_policy.end_exploration():
        memory = Memory()
        new_result = game.play_several_games(strategy=strategy, memory=memory, games_count=games_in_epoch)
        results.append(new_result)
        strategy.update_weights(memory=memory)
        exploration_policy.increment_epoch()
        if exploration_policy.current_epoch() % 10 == 0:
            with open(log_file_path, 'a') as log:
                date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(date_string + " epoch " + str(exploration_policy.current_epoch()) + "\n")

    return results


game = Game()
models = test_models(observation_space=game.observation_space(), action_space=game.action_space())

for index, model in enumerate(models):
    epochs = 1000
    start = time.time()
    results = train(game, model, epochs, 20)
    results = np.fromiter(map(lambda x: np.average(x), results), dtype=np.float)
    plt.clf()
    plt.plot(results)
    plt.savefig(artifacts_dir_path + model.name + ".png")
    model.save(artifacts_dir_path + model.name + ".h5")
