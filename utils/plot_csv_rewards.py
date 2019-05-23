# Author: Mikita Sazanovich

from collections import namedtuple
from typing import List
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Experiment = namedtuple('Experiment', ['name', 'params', 'filler', 'run_csvs'])

TOTAL_STEPS = 300000
BIN_SIZE = 30000
assert TOTAL_STEPS % BIN_SIZE == 0
TOTAL_BINS = TOTAL_STEPS // BIN_SIZE + 1


def plot_with_variance(values, params, filler, label):
    xs = np.arange(TOTAL_BINS) * BIN_SIZE
    ymeans = np.zeros(TOTAL_BINS)
    ystderrs = np.zeros(TOTAL_BINS)
    for i in range(TOTAL_BINS):
        xvalues = values[i]
        if len(xvalues) == 0:
            continue
        ymean = np.mean(xvalues)
        ystd = np.std(xvalues)
        ystderr = ystd / np.sqrt(len(xvalues))
        ymeans[i], ystderrs[i] = ymean, ystderr
    plt.plot(xs, ymeans, params, label=label)
    plt.fill_between(xs, ymeans - ystderrs, ymeans + ystderrs, color=filler, alpha=.2)


def init_the_plot():
    plt.figure(figsize=(10, 8))


def draw_the_plot(y_limit):
    fontsize = 16
    plt.xlabel('Шагов тренировки', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Награда', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis([None, None, 0, y_limit])
    plt.legend(loc='upper left', prop={'size': fontsize})
    plt.show()


def draw_mean_var_graphs(csv_dir: str, experiments: List[Experiment]) -> None:
    init_the_plot()
    for experiment in experiments:
        values = []
        for i in range(TOTAL_BINS):
            values.append([])
        for filename in experiment.run_csvs:
            csv_file = os.path.join(csv_dir, filename)
            with open(csv_file) as csv_file:
                df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                step = row['Step']
                if step >= TOTAL_STEPS:
                    continue
                value = row['Value']
                bin_id = int(step / BIN_SIZE) + 1
                if bin_id >= TOTAL_BINS:
                    continue
                values[bin_id].append(value)
        plot_with_variance(values, params=experiment.params, filler=experiment.filler,
                           label=experiment.name)
    draw_the_plot(y_limit=60)


def draw_smoothed_graphs(rewards_dir, filenames, labels):
    init_the_plot()
    smoothing_weight = 0.95
    rewards_dir = pathlib.Path(rewards_dir)
    for filename, label in zip(filenames, labels):
        csv_path = os.path.join(rewards_dir, filename)
        steps = []
        values = []
        last = None
        with open(str(csv_path), 'r') as csv_file:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                step = row['Step']
                if step > TOTAL_STEPS:
                    continue
                value = row['Value']
                if last is None:
                    last = value
                smoothed_value = last * smoothing_weight + (1 - smoothing_weight) * value
                last = smoothed_value
                steps.append(step)
                values.append(smoothed_value)
        plt.plot(steps, values, label=label)
    draw_the_plot(y_limit=80)


def main(draw_first, draw_second, draw_third):
    demo_filenames_dict = {
        0: [
            'run_20190504-no-demo-seed-50_summaries-tag-rewards.csv',
            'run_20190506-no-demo-seed-98_summaries-tag-rewards.csv',
            'run_20190506-no-demo-seed-79_summaries-tag-rewards.csv',
            'run_20190508-no-demo-seed-92_summaries-tag-rewards.csv',
        ],
        1: [
            'run_20190415-1-random-full-sigma-02_summaries-tag-rewards.csv',
            'run_20190416-1-random-full-sigma-02_summaries-tag-rewards.csv',
            'run_20190504-1-demo-1725_summaries-tag-rewards.csv',
            'run_20190505-1-demo-1738_summaries-tag-rewards.csv',
        ],
        50: [
            'run_20190419-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190421-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190429-50-demos-merged-seed-90_summaries-tag-rewards.csv',
            'run_20190503-50-demos-seed-32_summaries-tag-rewards.csv',
        ],
        150: [
            'run_20190423-150-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190425-150-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190425-150-demos-merged-sigma-02_summaries-tag-rewards.csv'
        ],
        254: [
            'run_20190413-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190417-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190418-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190507-all-merged-095-sigma-seed-32_summaries-tag-rewards.csv',
        ],
    }
    if draw_first:
        # Compiling the first paper graph
        draw_smoothed_graphs('/Users/niksaz/4-RnD/csv-results', [
            'run_20190419-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190420-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190421-50-demos-merged-sigma-02_summaries-tag-rewards.csv'
        ], [
            'Запуск 1',
            'Запуск 2',
            'Запуск 3',
        ])
    if draw_second:
        # Compiling the second paper graph
        draw_mean_var_graphs('/Users/niksaz/4-RnD/csv-results', [
            Experiment('Без демонстраций', 'k', 'black', demo_filenames_dict[0]),
            Experiment('С одной демонстрацией', 'm', 'magenta', demo_filenames_dict[1]),
        ])
    if draw_third:
        # Compiling the third paper graph
        draw_mean_var_graphs('/Users/niksaz/4-RnD/csv-results', [
            Experiment('С 1 демнострацией', 'm', 'magenta', demo_filenames_dict[1]),
            Experiment('С 50 демнострациями', 'y', 'yellow', demo_filenames_dict[50]),
            # Experiment('С 150 демнострациями', 'c', 'cyan', demo_filenames_dict[150]),
            Experiment('С 254 демнострациями', 'b', 'blue', demo_filenames_dict[254]),
            # Experiment('С 254 демонстрациями используя кластеризацию', 'r', 'red', [
            #     'run_20190427-cluster-all-1700_summaries-tag-rewards.csv',
            #     'run_20190428-cluster-all-1700_summaries-tag-rewards.csv',
            # ]),
        ])


if __name__ == '__main__':
    main(draw_first=True, draw_second=True, draw_third=True)
