#!/usr/bin/env python3
""" training the model """

from dqn import dqndef
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


dqn, env, args = dqndef()

# Okay, now it's time to learn something! We capture the interrupt
# exception so that training can be prematurely aborted.
# Notice that now you can use the built-in Keras callbacks!
weights_filename = 'policy.h5f'
checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(args.env_name)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                     interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)

# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=False)
