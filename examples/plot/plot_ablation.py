import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Column:
    STEP = 'Step'
    VALUE = 'Value'
    # ALGORITHM = 'Barrier Step'
    ALGORITHM = 'TTC'


class Algorithm:
    # ABL1 = '1'
    # ABL2 = '5'
    # ABL3 = '10'
    # ORIGINAL = '20'
    ABL1 = '(Inf, 0)'
    ABL2 = '(5, 0.5)'
    ORIGINAL = '(2, 1)'


class Criterion:
    COST = 'Average Episode Cost'
    RETURN = 'Average Episode Return'


algorithm_to_filename_pattern = {
    # Algorithm.ABL1: 'barrier1',
    # Algorithm.ABL2: 'barrier5',
    # Algorithm.ABL3: 'barrier10',
    Algorithm.ABL1: 'ttc0',
    Algorithm.ABL2: 'ttc0.5',
    Algorithm.ORIGINAL: 'original',
}

criterion_to_filename_pattern = {
    Criterion.COST: 'cost',
    Criterion.RETURN: 'reward',
}

algorithm_to_color = {
    Algorithm.ABL1: (20 / 255, 101 / 255, 200 / 255),
    Algorithm.ABL2: (30 / 255, 190 / 255, 30 / 255),
    # Algorithm.ABL3: (158 / 255, 10 / 255, 210 / 255),
    Algorithm.ORIGINAL: (220 / 255, 0, 0),
}


def load_data(path, criterion, algorithms):
    data = []
    for alg in algorithms:
        dirname = os.path.join(path, algorithm_to_filename_pattern[alg])
        filenames = os.listdir(dirname)
        for filename in filenames:
            if criterion_to_filename_pattern[criterion] not in filename:
                continue
            df = pd.read_csv(os.path.join(dirname, filename), sep=',')
            df[Column.VALUE] = smooth(df[Column.VALUE])
            df.insert(loc=len(df.columns), column=Column.ALGORITHM, value=alg)
            data.append(df)
    return pd.concat(data)


def smooth(x, width=5):
    y = np.ones(width)
    z = np.ones(len(x))
    return np.convolve(x, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    criterions = [Criterion.COST, Criterion.RETURN]
    algorithms = [
        Algorithm.ABL1,
        Algorithm.ABL2,
        # Algorithm.ABL3,
        Algorithm.ORIGINAL,
    ]

    for criterion in criterions:
        plt.figure(figsize=(4.5, 3.6))

        data = load_data('../../results/tensorboard_data/ablation', criterion, algorithms)

        sns.set(style="darkgrid", font_scale=1.0)
        legend = 'brief' if criterion == criterions[-1] else False
        sns.lineplot(data=data, x=Column.STEP, y=Column.VALUE, hue=Column.ALGORITHM, hue_order=algorithms,
                     ci=95, palette=algorithm_to_color, legend=legend)

        plt.xlim(data[Column.STEP].min(), data[Column.STEP].max())
        plt.xlabel('Environment Step')
        plt.ylabel(criterion)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.title('Point Goal')

        plt.tight_layout()
        plt.savefig(f'../../results/figures/ablation_{Column.ALGORITHM.replace(" ", "")}'
                    f'_{criterion.replace(" ", "")}.pdf')

    plt.show()
