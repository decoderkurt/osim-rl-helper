from helper.templates import Agent
import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI

class OpenAIAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, observation_space, action_space):
        # Parse noise_type
        self.action_noise = None
        self.param_noise = None
        nb_actions = action_space.shape[-1]
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))

        # Configure components.
        self.memory = Memory(limit=int(1e6), action_shape=action_space.shape, observation_shape=observation_space.shape)
        self.critic = Critic(layer_norm=True)
        self.actor = Actor(nb_actions, layer_norm=True)

    def train(self, env, nb_steps):
        training.train(env=env, param_noise=self.param_noise,
            action_noise=self.action_noise, actor=self.actor, critic=self.critic, memory=self.memory,
            nb_epochs=500, nb_epoch_cycles=20, render_eval=False, reward_scale=1.0, render=False, normalize_returns=False, normalize_observations=True, critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3, popart=False, gamma=0.99, clip_norm=None, nb_train_steps=nb_steps, nb_rollout_steps=100, nb_eval_steps=100, batch_size=64)

