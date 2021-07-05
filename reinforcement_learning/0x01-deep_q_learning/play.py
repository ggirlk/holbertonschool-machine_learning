from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Conv2D, Reshape
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

class args:
    env_name = "Breakout-v0" # 'BreakoutDeterministic-v4'
    mode = 'play'
    weights = None

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = Flatten()(layer3)

    layer5 = Dense(512, activation="relu")(layer4)
    action = Dense(nb_actions, activation="linear")(layer5)

    return Model(inputs=inputs, outputs=action)

model = create_q_model()

print(model.summary())

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)


dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4,
               delta_clip=1.)

weights_filename = 'policy.h5f'.format(args.env_name)
if args.weights:
    weights_filename = args.weights
dqn.load_weights(weights_filename)
dqn.test(env, nb_episodes=10, visualize=True)
