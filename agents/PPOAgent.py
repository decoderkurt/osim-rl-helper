from helper.templates import Agent
import numpy as np
from baselines import bench, logger
from baselines.common import tf_util as U
import os
from mpi4py import MPI
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class PPOAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space

    def train(self, env, nb_steps):
            from baselines.ppo1 import mlp_policy, pposgd_simple
            U.make_session(num_cpu=1).__enter__()
            def policy_fn(name, ob_space, ac_space):
                return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                    hid_size=64, num_hid_layers=2)
            pposgd_simple.learn(env, policy_fn,
                    max_timesteps=int(1e6),
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                    gamma=0.99, lam=0.95, schedule='linear',
                )
