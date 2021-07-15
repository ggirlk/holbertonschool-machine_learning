#!/usr/bin/env python3
""" SARSA(λ) """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    ******************************************************
    ***********uses epsilon-greedy to determine***********
    *******************the next action********************
    ******************************************************
    @Q: is a numpy.ndarray containing the q-table
    @state: is the current state
    @epsilon: is the epsilon to use for the calculation
    *** You should sample p with numpy.random.uniformn to determine
        if your algorithm should explore or exploit
    *** If exploring, you should pick the next action with
        numpy.random.randint from all possible actions
    Returns:
            the next action
    """
    p = np.random.uniform()
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    **********************************************************************
    ***************************perform SARSA(λ)***************************
    **********************************************************************
    @env: is the openAI environment instance
    @Q: is a numpy.ndarray of shape (s,a) containing the Q table
    @lambtha: is the eligibility trace factor
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate
    @epsilon: is the initial threshold for epsilon greedy
    @min_epsilon: is the minimum value that epsilon should decay to
    @epsilon_decay: is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    for episode in range(episodes):
        # Reseting the environment
        state = env.reset()
        # Taking action
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for step in range(max_steps):
            # Getting the reward and outcome state
            new_state, reward, done, info = env.step(action)
            # Taking new action
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            # Updating Q table
            Q[state, action] = (((1 - lambtha) * Q[state, action])
                                + alpha * (reward + gamma
                                           * Q[new_state, new_action]
                                           - Q[state, action]))
        if done:
            break
        # Incrementing state and action
        state = new_state
        action = new_action
    # Returning the updated Q table
    return Q
