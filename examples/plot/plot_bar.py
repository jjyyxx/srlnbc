import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Column:
    STEP = 'Step'
    VALUE = 'Value'
    ALGORITHM = 'Algorithm'
    CRITERION = 'Criterion'


class Algorithm:
    PPO_LAGRANGIAN = 'PPO-Lagrangian'
    TRPO_LAGRANGIAN = 'TRPO-Lagrangian'
    CPO = 'CPO'
    BARRIER = 'PPO-Barrier (ours)'


class Criterion:
    COST = 'Safety Violation'
    SUCCESS = 'Performance'


class Env:
    POINT_GOAL = 'Point Goal'
    CAR_GOAL = 'Car Goal'
    METADRIVE = 'MetaDrive'


algorithm_to_filename_pattern = {
    Algorithm.PPO_LAGRANGIAN: 'ppo',
    Algorithm.TRPO_LAGRANGIAN: 'trpo',
    Algorithm.CPO: 'cpo',
    Algorithm.BARRIER: 'Barrier',
}

criterion_to_filename_pattern = {
    Criterion.COST: 'cost',
    Criterion.SUCCESS: 'success',
}

env_to_filename_pattern = {
    Env.POINT_GOAL: 'point_goal',
    Env.CAR_GOAL: 'car_goal',
    Env.METADRIVE: 'metadrive',
}


def load_data(path, algorithms):
    data = []
    filenames = os.listdir(path)
    for alg in algorithms:
        for filename in filenames:
            if algorithm_to_filename_pattern[alg] in filename:
                df = pd.read_csv(os.path.join(path, filename), sep=',')
                df[Column.VALUE] = smooth(df[Column.VALUE])
                value = df.iloc[-1].at[Column.VALUE]
                data.append({
                    Column.ALGORITHM: alg,
                    Column.VALUE: value,
                })
    return pd.DataFrame(data)


def smooth(x, width=5):
    y = np.ones(width)
    z = np.ones(len(x))
    return np.convolve(x, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    env = Env.METADRIVE
    criterion = Criterion.COST
    algorithms = [
        Algorithm.PPO_LAGRANGIAN,
        Algorithm.TRPO_LAGRANGIAN,
        Algorithm.CPO,
        Algorithm.BARRIER
    ]
    legend = False

    data_path = os.path.join(
        '../../results/tensorboard_data/comparison',
        env_to_filename_pattern[env],
        criterion_to_filename_pattern[criterion]
    )
    data = load_data(data_path, algorithms)

    sns.set(style="darkgrid", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6))
    sns.barplot(data=data, x=Column.ALGORITHM, y=Column.VALUE,
                ci='sd', capsize=0.2, errwidth=2)
    plt.ylim(0)
    plt.xticks(rotation=15)
    plt.xlabel(None)
    plt.ylabel(criterion)
    plt.tight_layout()

    plt.savefig(f'../../results/figures/Barplot_{env.replace(" ", "")}_{criterion.replace(" ", "")}.pdf')
    plt.show()
