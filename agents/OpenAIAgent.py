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
import agents.OpenAITraining as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
#import cloudpickle
#from baselines.common import tf_util as U
#from baselines.common.tf_util import load_state, save_state

import gym
import tensorflow as tf
from mpi4py import MPI

class OpenAIAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, observation_space, action_space):
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure()
        self.seed  = 0

    def train(self, env, nb_steps):
        # Configure things.
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)

        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = env.action_space.shape[-1]
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))

        # Configure components.
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
        critic = Critic(layer_norm=True)
        actor = Actor(nb_actions, layer_norm=True)

        # Seed everything to make things reproducible.
        seed = self.seed + 1000000 * rank
        logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
        tf.reset_default_graph()
        set_global_seeds(seed)
        env.seed(seed)

        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()
        #load_state("D:\project\osim-rl-helper\ddpg.pkl")
        training.train(env=env, param_noise=param_noise,restore=True,
            action_noise=action_noise, actor=actor, critic=critic, memory=memory,
            nb_epochs=1, nb_epoch_cycles=1, render_eval=False, reward_scale=1.0, render=False, normalize_returns=False, normalize_observations=True, critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3, popart=False, gamma=0.99, clip_norm=None, nb_train_steps=nb_steps, nb_rollout_steps=5, nb_eval_steps=5, batch_size=64)
        #save_state("D:\project\osim-rl-helper\ddpg.pkl")

        if rank == 0:
            logger.info('total runtime: {}s'.format(time.time() - start_time))
