#!/usr/bin/env python3
""" Monte Carlo """
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    ***********************************************************
    *************perform the Monte Carlo algorithm*************
    ***********************************************************
    @env: is the openAI environment instance
    @V: is a numpy.ndarray of shape (s,) containing the value estimate
    @policy: is a function that takes in a state and returns
             the next action to take
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate
    Returns: V, the updated value estimate
    """

    for ep in range(episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Create Cast and turn episode list to np.ndarray
        episode = [[0, 0, 0, 0]] * max_steps
        episode = np.array(episode, dtype=int)
        # initiate needed variabes
        T = len(episode)  # total number of states starting from 0
        G = 0  # empirical return
        for step in range(max_steps):
            t = step
            # taking action
            action = policy(state)
            # Taking the action and getting the reward and outcome state
            new_state, Returns, done, info = env.step(action)
            # append results for each state of episode
            episode[step] = state, action, Returns, new_state
            # end of game
            if done:
                break
            # calculate empirical return
            G += gamma**t * Returns  # summing returns (rewards)
            # Value Estimation
            if state not in episode[:ep, 0]:
                V[state] = V[state] + alpha * (G - V[state])
            # Incrementing the state
            state = new_state
    # Returning the updated Value Estimate
    return V
