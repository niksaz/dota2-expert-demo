# Author: Mikita Sazanovich

from collections import namedtuple
from typing import List
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Experiment = namedtuple('Experiment', ['name', 'color', 'run_csvs'])

TOTAL_STEPS = 300000
BIN_SIZE = 30000
assert TOTAL_STEPS % BIN_SIZE == 0
TOTAL_BINS = TOTAL_STEPS // BIN_SIZE + 1


def plot_with_variance(values, color, label):
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
    plt.plot(xs, ymeans, color, label=label)
    plt.fill_between(xs, ymeans - ystderrs, ymeans + ystderrs, color=color, alpha=.2)


def init_the_plot():
    plt.figure(figsize=(10, 8))


def draw_the_plot(y_limit):
    fontsize = 16
    plt.xlabel('Training Steps', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Episode Discounted Reward', fontsize=fontsize)
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
        plot_with_variance(values, color=experiment.color, label=experiment.name)
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


def verify_results_existence(results_dir, id_filenames_dict):
    for filenames in id_filenames_dict.values():
        for filename in filenames:
            result_filename = os.path.join(results_dir, filename)
            assert os.path.exists(result_filename)


def main():
    results_dir = '/Users/niksaz/4-RnD/csv-results'
    id_filenames_dict = {
        '0-demo': [
            'run_20190504-no-demo-seed-50_summaries-tag-rewards.csv',
            'run_20190506-no-demo-seed-98_summaries-tag-rewards.csv',
            'run_20190506-no-demo-seed-79_summaries-tag-rewards.csv',
            'run_20190508-no-demo-seed-92_summaries-tag-rewards.csv',
        ],
        '1-demo': [
            'run_20190415-1-random-full-sigma-02_summaries-tag-rewards.csv',
            'run_20190416-1-random-full-sigma-02_summaries-tag-rewards.csv',
            'run_20190504-1-demo-1725_summaries-tag-rewards.csv',
            'run_20190505-1-demo-1738_summaries-tag-rewards.csv',
        ],
        '50-demos': [
            'run_20190419-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190421-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190429-50-demos-merged-seed-90_summaries-tag-rewards.csv',
            'run_20190503-50-demos-seed-32_summaries-tag-rewards.csv',
        ],
        '150-demos': [
            'run_20190423-150-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190425-150-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190425-150-demos-merged-sigma-02_summaries-tag-rewards.csv'
        ],
        '254-demos': [
            'run_20190413-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190417-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190418-all-merged-095-sigma-02_summaries-tag-rewards.csv',
            'run_20190507-all-merged-095-sigma-seed-32_summaries-tag-rewards.csv',
        ],
        'clustering': [
            'run_20190427-cluster-all-1700_summaries-tag-rewards.csv',
            'run_20190428-cluster-all-1700_summaries-tag-rewards.csv',
        ],
        'stochastic': [
            'run_20190419-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190420-50-demos-merged-sigma-02_summaries-tag-rewards.csv',
            'run_20190421-50-demos-merged-sigma-02_summaries-tag-rewards.csv'
        ],
        'random': [
            'run-20191009-random-seed-7_summaries-tag-rewards.csv',
            'run-20191010-random-seed-46_summaries-tag-rewards.csv',
            'run-20191011-random-seed-10_summaries-tag-rewards.csv',
            'run-20191015-random-seed-78_summaries-tag-rewards.csv',
        ],
    }
    verify_results_existence(results_dir, id_filenames_dict)

    # Compiling the stochasticity visual proof graph
    draw_smoothed_graphs(results_dir, id_filenames_dict['stochastic'], [
        'Run 1',
        'Run 2',
        'Run 3',
    ])
    # Compiling the demo/no-demo graph
    draw_mean_var_graphs(results_dir, [
        Experiment('No demonstrations', 'black', id_filenames_dict['0-demo']),
        Experiment('Single demonstration', 'magenta', id_filenames_dict['1-demo']),
    ])
    # Compiling the comparison of random sampling with our approach
    draw_mean_var_graphs(results_dir, [
        Experiment('Our approach', '#F5B041', id_filenames_dict['50-demos']),
        Experiment('Random sampling', '#708090', id_filenames_dict['random'])
    ])
    # Compiling the different volumes graph
    draw_mean_var_graphs(results_dir, [
        Experiment('With 1 demonstration', 'magenta', id_filenames_dict['1-demo']),
        Experiment('With 50 demonstrations', '#F5B041', id_filenames_dict['50-demos']),
        # Experiment('With 150 demonstrations', 'cyan', id_filenames_dict['150-demos']),
        Experiment('With 254 demonstrations', 'blue', id_filenames_dict['254-demos']),
        # Experiment('With 254 demonstrations clustered', 'red', id_filenames_dict['clustering']),
    ])


if __name__ == '__main__':
    main()
