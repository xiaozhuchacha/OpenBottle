
import numpy as np
import scipy.io
import glob
import os
import csv

if __name__ == '__main__':

    index_name = ['end', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                 'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    frequencies = np.zeros((len(index_name), len(index_name)))

    mat_files = glob.glob('data/*.mat')
    for mat_file in mat_files:
        data = scipy.io.loadmat(mat_file)
        data = data['windows']
        tokens = mat_file.replace('/', '_').split('_')
        if len(tokens) > 3:
            current_action = tokens[1] + '_' + tokens[2]
        else:
            current_action = tokens[1]
        current_action_idx = index_name.index(current_action)
        next_action_labels = np.array([max(0, x) for x in data[0::2, -1]])
        for i in range(len(next_action_labels)):
            next_action_idx = int(next_action_labels[i])

            frequencies[current_action_idx][next_action_idx] += 1

    freq_sums = frequencies.sum(axis=1)
    probabilities = frequencies / freq_sums[:, np.newaxis]

    ind = np.isnan(probabilities)
    probabilities[ind] = 0

    scipy.io.savemat('prior_counts.mat', {'priors': probabilities})
