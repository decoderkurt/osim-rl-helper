from .Wrapper import EnvironmentWrapper
import gym
import numpy as np


class ForceDictObservation(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment wrapper that wraps local environment to use dict-type
        observation by setting project=False. This can be deprecated once
        the default observation is dict-type rather than list-type.
        """
        super().__init__(env)
        self.env = env
        self.time_limit = 300
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(19, ),
                                           dtype=np.float32)

    def reset(self):
        return self.env.reset(project=False)

    def step(self, action):
        return self.env.step(action, project=False)
