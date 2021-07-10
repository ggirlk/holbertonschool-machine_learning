#!/usr/bin/env python3
""" visualizing the game """

from dqn import dqndef


dqn, env, _ = dqndef()

weights_filename = 'policy.h5f'
dqn.load_weights(weights_filename)
dqn.test(env, nb_episodes=10, visualize=True)
