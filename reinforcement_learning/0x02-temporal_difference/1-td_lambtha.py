#!/usr/bin/env python3
""" TD(λ) Off-line λ-return algorithm """
import numpy as np


def td_lambtha(env, V, policy, lambtha=1, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    ************************************************************
    ****************performs the TD(λ) algorithm****************
    ************************************************************
    @env: is the openAI environment instance
    @V: is a numpy.ndarray of shape (s,) containing the value estimate
    @policy: is a function that takes in a state and returns the next
             action to take
    @lambtha: is the eligibility trace factor
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate
    Returns: V, the updated value estimate
    """
    for ep in range(episodes):
        # Reseting the environment
        state = env.reset()
        # Create Cast and turn episode list to np.ndarray
        episode = [[0, 0, 0, 0]] * max_steps
        episode = np.array(episode, dtype=int)
        # initiate needed variabes
        T = len(episode)  # total number of states starting from 0
        G = 0  # empirical return
        n = 1  # number of steps
        Gtn = 0
        Gtnlamda = 0
        for step in range(max_steps):
            t = step
            # Taking action
            action = policy(state)
            # Getting the reward and outcome state
            new_state, Returns, done, info = env.step(action)
            # Appending results for each state of episode
            episode[step] = state, action, Returns, new_state
            # end of game
            if done:
                break
            # calculate Gt de n step and sum it
            G += gamma**t * Returns  # summing returns (rewards)
            Gtn += (G + gamma**(n) * V[new_state]) * lambtha**(n - 1)
            # calculate Gtn lambda by weights decay:
            #                             a factor λ with n,  λ^(n−1)
            Gtnlamda = (1 - lambtha) * Gtn  # λ-return
            # Value Estimation
            if state not in episode[:ep, 0]:
                # V[state] = (1 - alpha) * V[state] + alpha * Gtnlamda
                V[state] = V[state] + alpha * (Gtnlamda - V[state])
                # V[state] = V[state] + alpha * (Returns + gamma
                #                                * V[new_state] - V[state])
            n += 1
            # Incrementing the state
            state = new_state
    # Returning the updated Value Estimate
    return V
