# --------------------------------------------------------
# Particle Filter
# Licensed under The MIT License
# Written by Ye Liu (ye-liu at whu.edu.cn)
# --------------------------------------------------------

import torch
import scipy.stats
import numpy as np

from utils import neff


class ParticleFilter(object):
    def __init__(self, num_particles, particle_sampling_method, x_range, y_range, means, stds, motion_std_err, measurement_std_err, num_predict_steps,
                 plot_particles):
        self._num_particles = num_particles
        self._motion_std_err = motion_std_err
        self._measurement_std_err = measurement_std_err
        self._num_predict_steps = num_predict_steps
        self._plot_particles = plot_particles

        self._actual_pts = []
        self._estimate_pts = []

        self._init_particles(particle_sampling_method, x_range, y_range, means, stds)

    @property
    def actual_pts(self):
        return self._actual_pts

    @property
    def estimate_pts(self):
        return self._estimate_pts

    @property
    def particles(self):
        return self._particles[:, :2]

    def _init_particles(self, particle_sampling_method, x_range, y_range, means, stds):
        # two methods to initialize states of particles
        def init_uniform_states(x_range, y_range):
            states = torch.empty(self._num_particles, 8)
            states[:, 0].uniform_(x_range[0], x_range[1])  # x axis
            states[:, 1].uniform_(y_range[0], y_range[1])  # y axis
            states[:, 2:4].uniform_(-1, 1)  # speed
            states[:, 4:6].uniform_(-0.1, 0.1)  # acceration
            states[:, 6:].uniform_(-0.1, 0.1)  # second-order acceration
            return states

        def init_gaussian_states(means, stds):
            return torch.normal(means.expand(self._num_particles, -1), stds.expand(self._num_particles, -1))

        # initialize states from a distribution
        if particle_sampling_method == 'UNIFORM':
            states = init_uniform_states(x_range, y_range)
        elif particle_sampling_method == 'GAUSSIAN':
            states = init_gaussian_states(means, stds)
        else:
            raise ValueError("Argument 'particle_sampling_method' must be 'UNIFORM' or 'GAUSSIAN'")

        # create particles with equal weights
        weights = torch.FloatTensor((1 / self._num_particles, ))
        self._particles = torch.cat((states, weights.expand(self._num_particles, -1)), dim=1)

    def _motion_update(self, particles, stds, dt):
        noisy_speed = particles[:, 2:4] + torch.randn(self._num_particles, 2) * stds[0]
        noisy_acc = particles[:, 4:6] + torch.randn(self._num_particles, 2) * stds[1]
        noisy_acc_acc = particles[:, 6:8] + torch.randn(self._num_particles, 2) * stds[2]
        particles[:, :2] += noisy_speed * dt + 0.5 * noisy_acc * dt**2
        particles[:, 2:4] = noisy_speed + noisy_acc * dt
        particles[:, 4:6] = noisy_acc + noisy_acc_acc * dt
        return particles

    def _measurement_update(self, pt, std):
        dist = torch.norm(self._particles[:, :2] - pt[:2], dim=1)
        self._particles[:, -1] = torch.FloatTensor(scipy.stats.norm(0, std).pdf(dist + torch.randn(1) * std))
        self._particles[:, -1] += 1.e-300  # avoid round-off to zero
        self._particles[:, -1] /= sum(self._particles[:, -1])  # normalize

    def _systematic_resample(self):
        positions = (torch.rand(1) + torch.arange(self._num_particles).float()) / self._num_particles
        indexes = torch.empty(self._num_particles)
        cumulative_sum = torch.cumsum(self._particles[:, -1], dim=0)
        i, j = 0, 0
        while i < self._num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def _resample_from_index(self, indexes):
        self._particles = self._particles[indexes.long()]
        self._particles[:, -1].new_full((self._num_particles, 1), 1.0 / self._num_particles)

    def _estimate(self, particles):
        mean = np.average(particles[:, :2].numpy(), weights=particles[:, -1].numpy(), axis=0)
        # var = np.average((particles[:, :2] - mean)**2, weights=particles[:, -1], axis=0)
        return mean

    def _predict(self, num_predict_steps):
        predict_pts = []
        particles = self._particles.clone()
        for _ in range(num_predict_steps):
            particles = self._motion_update(particles, self._motion_std_err, 1)
            mean = self._estimate(particles)
            predict_pts.append(mean)
        return predict_pts

    def update_state(self, pt):
        self._actual_pts.append(pt)
        pt = torch.FloatTensor(pt)

        # predict next position
        self._particles = self._motion_update(self._particles, self._motion_std_err, 1)

        # incorporate measurements
        self._measurement_update(pt, self._measurement_std_err)

        # resample if too few effective particles
        if neff(self._particles[:, -1]) < self._num_particles / 2:
            indexes = self._systematic_resample()
            self._resample_from_index(indexes)

        # estimate position
        mu = self._estimate(self._particles)

        # save results

        self._estimate_pts.append(mu)

        # predict
        predict_pts = self._predict(self._num_predict_steps)

        return predict_pts
