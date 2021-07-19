#!/usr/bin/env python3
""" Policy Gradients """
import numpy as np


def policy(state, weight):
    """
    ****************************************************
    ****computes to policy with a weight of a matrix****
    ****************************************************
    @state: matrix representing the current
            observation of the environment
    @weight: matrix of random weight
    Return: weighted policy
    """
    # calculate using softmax
    z = np.dot(state, weight)
    ez = np.exp(z)
    ez /= ez.sum()
    return ez


def policy_gradient(state, weight):
    """
    ****************************************************
    ******computes the Monte-Carlo policy gradient******
    ********based on a state and a weight matrix********
    ****************************************************
    @state: matrix representing the current
            observation of the environment
    @weight: matrix of random weight
    """
    state = state.reshape(1, -1)
    policy_value = policy(state, weight)
    action = np.random.choice((policy_value[0]).shape[0], p=policy_value[0])
    # ∇lnπ(At|St,θ)
    # left = (∇θ)log(e^(ϕ(s,a).⊺*θ)) = (∇θ)ϕ(s,a).⊺*θ = ϕ(s,a)
    # right = ∑k=1toN (ϕ(s,ak) * (πθ)(s,ak))
    # (∇θ)log((πθ)(s,a)) = left − right = ϕ(s,a) − (Eπθ)[ϕ(s,⋅)]
    grad = state.T - (state.T * policy_value[:, None]).sum(axis=0)
    return action, grad
