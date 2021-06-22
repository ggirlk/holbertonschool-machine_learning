#!/usr/bin/env python3
""" Play """
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """
    *********************************************
    ********trained agent play an episode********
    *********************************************
    @env: is the FrozenLakeEnv instance
    @Q: is a numpy.ndarray containing the Q-table
    @max_steps: is the maximum number of steps in
                the episode
    *** Each state of the board should be displayed
        via the console
    *** always exploit the Q-table
    Returns:
            the total rewards for the episode
    """
    # Reseting the environment
    state = 0
    env.reset()
    env.render()
    for step in range(max_steps):
        # Performing epsilon greedy
        action = epsilon_greedy(Q, state, 0)
        # Taking the action and getting the reward and outcome state
        state, reward, done, info = env.step(action)
        env.render()
        # Agent falling in a hole
        if done and reward == 0:
            return reward
        # Ending the episode
        if done:
            return reward
