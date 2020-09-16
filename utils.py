# --------------------------------------------------------
# Particle Filter
# Licensed under The MIT License
# Written by Ye Liu (ye-liu at whu.edu.cn)
# --------------------------------------------------------

import numpy as np
import torch
from matplotlib import pyplot as plt


def get_default_hyperparams():
    return {
        # particle filter initialization
        'num_particles': 5000,

        # particle sampling
        'sampling_method': 'uniform',
        'x_range': (-100, 100),
        'y_range': (-100, 100),
        'means': None,
        'stds': None,

        # motion update (speed, acceleration, second-order acceleration)
        'motion_std_err': (1, 0.5, 0.5),

        # measurement update
        'measurement_std_err': 1,

        # predict
        'num_predict_steps': 5,

        # plotting
        'plot_particles': True
    }


def neff(w):
    return 1.0 / torch.sum(torch.pow(w, 2))


def plot(actual_pts, estimate_pts, predict_pts, particles=None):
    actual_pts = np.array(actual_pts)
    estimate_pts = np.array(estimate_pts)
    predict_pts = np.array(predict_pts)

    plt.figure()

    if particles is not None:
        p1 = plt.scatter(
            particles[:, 0], particles[:, 1], alpha=0.2, color='g')

    p2 = plt.scatter(
        actual_pts[:, 0], actual_pts[:, 1], marker='+', color='k', s=100, lw=3)
    p3 = plt.scatter(estimate_pts[:, 0], estimate_pts[:, 1], color='r')
    p4 = plt.scatter(predict_pts[:, 0], predict_pts[:, 1], color='b')

    if particles is not None:
        plt.legend([p1, p2, p3, p4],
                   ['Particles', 'Actual', 'Estimate', 'Predict'],
                   loc=4,
                   numpoints=1)
    else:
        plt.legend([p2, p3, p4], ['Actual', 'Estimate', 'Predict'],
                   loc=4,
                   numpoints=1)

    plt.show()
