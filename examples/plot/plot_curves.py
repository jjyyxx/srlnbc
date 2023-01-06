import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


class Column:
    STEP = 'Step'
    VALUE = 'Value'
    ALGORITHM = 'Algorithm'
    CRITERION = 'Criterion'


class Algorithm:
    PPO_LAGRANGIAN = 'PPO-Lagrangian'
    TRPO_LAGRANGIAN = 'TRPO-Lagrangian'
    CPO = 'CPO'
    FAC_SIS = 'FAC-SIS'
    BARRIER = 'PPO-Barrier (ours)'


class Criterion:
    COST = 'Average Episode Cost'
    RETURN = 'Average Episode Return'
    SUCCESS = 'Average Success Rate'


class Env:
    POINT_GOAL = 'Point Goal'
    CAR_GOAL = 'Car Goal'
    METADRIVE = 'MetaDrive'


algorithm_to_filename_pattern = {
    Algorithm.PPO_LAGRANGIAN: 'ppo',
    Algorithm.TRPO_LAGRANGIAN: 'trpo',
    Algorithm.CPO: 'cpo',
    Algorithm.FAC_SIS: 'fac-sis',
    Algorithm.BARRIER: 'Barrier',
}

criterion_to_filename_pattern = {
    Criterion.COST: 'cost',
    Criterion.RETURN: 'reward',
    Criterion.SUCCESS: 'success',
}

env_to_filename_pattern = {
    Env.POINT_GOAL: 'point_goal',
    Env.CAR_GOAL: 'car_goal',
    Env.METADRIVE: 'metadrive',
}

algorithm_to_color = {
    Algorithm.PPO_LAGRANGIAN: (158 / 255, 10 / 255, 210 / 255),
    Algorithm.TRPO_LAGRANGIAN: (20 / 255, 101 / 255, 200 / 255),
    Algorithm.CPO: (30 / 255, 190 / 255, 30 / 255),
    Algorithm.FAC_SIS: (205 / 255, 164 / 255, 15 / 255),
    Algorithm.BARRIER: (220 / 255, 0, 0),
}


def load_data(path, algorithms):
    data = {alg: [] for alg in algorithms}
    filenames = os.listdir(path)
    for filename in filenames:
        for alg in algorithms:
            if algorithm_to_filename_pattern[alg] in filename:
                df = pd.read_csv(os.path.join(path, filename), sep=',')
                df = df.drop(labels='Wall time', axis=1)
                df[Column.VALUE] = smooth(df[Column.VALUE])
                df.insert(loc=len(df.columns), column=Column.ALGORITHM, value=alg)
                data[alg].append(df)
                break
    dfs = []
    for df in data.values():
        dfs.extend(df)
    data = pd.concat(dfs, ignore_index=True)
    return data


def smooth(x, width=5):
    y = np.ones(width)
    z = np.ones(len(x))
    return np.convolve(x, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    env = Env.CAR_GOAL
    criterion = Criterion.COST
    algorithms = [
        Algorithm.PPO_LAGRANGIAN,
        Algorithm.TRPO_LAGRANGIAN,
        Algorithm.CPO,
        Algorithm.FAC_SIS,
        Algorithm.BARRIER
    ]
    legend = False
    magnifier = True

    data_path = os.path.join(
        '../../results/tensorboard_data/comparison',
        env_to_filename_pattern[env],
        criterion_to_filename_pattern[criterion]
    )
    data = load_data(data_path, algorithms)

    sns.set(style="darkgrid", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.6))
    sns.lineplot(data=data, x=Column.STEP, y=Column.VALUE, hue=Column.ALGORITHM, ci=95,
                 palette=algorithm_to_color, legend=legend)
    plt.xlim(data[Column.STEP].min(), data[Column.STEP].max())
    plt.xlabel('Environment Step')
    plt.ylabel(criterion)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(env)
    plt.tight_layout()

    if magnifier:
        axins = inset_axes(ax, width='40%', height='30%', loc='lower left',
                           bbox_to_anchor=(0.5, 0.6, 1, 1),
                           bbox_transform=ax.transAxes)
        sns.lineplot(data=data, x=Column.STEP, y=Column.VALUE, hue=Column.ALGORITHM, ci=95,
                     palette=algorithm_to_color, legend=legend)
        plt.xlim(data[Column.STEP].max() * 0.9, data[Column.STEP].max())
        plt.ylim(-0.5, 0.5)
        plt.xlabel(None)
        plt.ylabel(None)
        mark_inset(ax, axins, loc1=3, loc2=4, fc='none', ec='k', lw=1)

    plt.savefig(f'../../results/figures/{env.replace(" ", "")}_{criterion.replace(" ", "")}.pdf')
    plt.show()
