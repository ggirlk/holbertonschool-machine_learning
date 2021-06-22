#!/usr/bin/env python3
""" Q-learning """
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    ***********************************************
    **************performs Q-learning**************
    ***********************************************
    @env: is the FrozenLakeEnv instance
    @Q: is a numpy.ndarray containing the Q-table
    @episodes: is the total number of episodes to train over
    @max_steps: is the maximum number of steps per episode
    @alpha: is the learning rate
    @gamma: is the discount rate
    @epsilon: is the initial threshold for epsilon greedy
    @min_epsilon: is the minimum value that epsilon should
                  decay to
    @epsilon_decay: is the decay rate for updating epsilon
                    between episodes
    *** When the agent falls in a hole, the reward should
        be updated to be -1
    Returns:
            Q: is the updated Q-table
            total_rewards: is a list containing the rewards
                           per episode
    """
    training_rewards = []
    epsilons = []
    max_epsilon = 1
    for episode in range(episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # Performing epsilon greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)

            # Agent falling in a hole
            if done and reward == 0:
                reward = -1

            # Updating the Q-table using the Bellman equation
            Q[state, action] = (Q[state, action] + alpha
                                * (reward + gamma * np.max(Q[new_state])
                                - Q[state, action]))

            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done:
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = (min_epsilon + (max_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    return Q, training_rewards
