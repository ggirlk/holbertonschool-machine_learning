#!/usr/bin/env python3
""" DQNAgent """

from __future__ import division

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, Flatten, Permute, Conv2D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy,\
                      EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        # (height, width, channel)
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`.
        # In this case, however,
        # we would need to store a `float32` array instead, which is 4x
        # more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


class args:
    env_name = 'Breakout-v0'
    mode = ''
    weights = None


def dqndef():
    # Get the environment and extract the number of actions.
    env = gym.make(args.env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build our model. We use the same model that was
    # described by Mnih et al. (2015).
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    print(model.summary())
    # print(model.output_shape)
    # Finally, we configure and compile our agent. You can use
    # every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    # Select a policy. We use eps-greedy action selection, which means that
    # a random action is selected with probability eps.
    # We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done
    # so that the agent initially explores the environment (high eps) and
    # then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing.
    # Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures
    # that the agent cannot get stuck.

    policy = GreedyQPolicy()

    # The trade-off between exploration and exploitation is difficult
    # and an on-going research topic.
    # If you want, you can experiment with the parameters or use a
    # different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
    #                                  value_max=1., value_min=.1,
    #                                  value_test=.05, b_steps=1000000)
    # Feel free to give it a try!

    # print(model.output_shape)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   memory=memory, processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    return dqn, env, args
