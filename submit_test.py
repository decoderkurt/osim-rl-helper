import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv
import numpy as np
import argparse

# Settings
remote_base = 'http://grader.crowdai.org:1729'

token = 'c4cda3976f22b8f468b78a33e47bb432'

client = Client(remote_base)

# Create environment
observation = client.env_create(token, env_id="ProstheticsEnv")
env = ProstheticsEnv(visualize=False)

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

# Run a single step
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    #print(observation)
    env.action = [0, 0, 0.001, 0.001, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    [observation, reward, done, info] = client.env_step(env.action)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()