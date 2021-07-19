#!/usr/bin/env python3
""" Policy Gradients """
import numpy as np

from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    *********************************************
    ******Implementation of a full training******
    *********************************************
    @env: initial environment
    @nb_episodes: number of episodes used for training
    @alpha: the learning rate
    @gamma: the discount factor
    Return:
        all values of the score (sum of all rewards
        during one episode loop)
    """
    # Inisiate scores list
    scores = []
    # Initiate θ to random
    # np.random.seed(0)
    # env.seed(0)
    W = np.random.rand(env.observation_space.shape[0],
                       env.action_space.n)
    for ep in range(nb_episodes):
        # **** Generating episode *****************************
        # Reseting the environment each time as per requirement
        state = env.reset()[None, :]
        # initiate needed variabes
        done = False
        t = 0
        R = []
        Grads = []
        Actions = []
        while not done:
            # Renderig the environment every 1000
            if show_result and not ep % 1000:
                env.render()
            # Taking action and gradient
            action, grad = policy_gradient(state, W)
            # Getting the reward and outcome state
            new_state, Returns, done, info = env.step(action)
            # Appending needed Values
            Actions.append(action)
            R.append(Returns)
            Grads.append(grad)
            # Incrementing state
            state = new_state[None, :]
            t += 1
        # Appending summed score
        scores.append(sum(R))
        print("Episode N°: " + str(ep) + " Score: " + str(sum(R)),
              end="\r", flush=False)

        # **** Updating θ ***************************************************
        # initiate needed variabes
        G = 0  # empirical return
        T = t
        for t in range(T):
            Returns = R[t]
            action = Actions[t]
            # Gt = ∑k=0 to ∞ (γ^(k) * R(t+k+1))
            G = sum(gamma**(k) * R[k+t+1] for k in range(T-t-1))
            # θ ← θ + α * γ^(t) * Gt * ∇θlnπθ(At|St) ; from Barto Satton book
            # W[:, action] += alpha * Grads[t][:, action] * gamma**(t) * G
            # θ ← θ + α * ∇θlogπθ(st, at) * vt ; from David Silver course
            W[:, action] += alpha * Grads[t][:, action] * G

    return scores
