#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """ doc """
    callbacks = None
    if validation_data:
        callbacks = [K.callbacks.EarlyStopping(monitor="loss",
                                               patience=patience,
                                               mode="auto")]

        def scheduler(epoch):
            """ new learning """
            return alpha/(1+(decay_rate*(epoch)))
        callbacks.append(K.callbacks.LearningRateScheduler(scheduler, 1))
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=callbacks,
                       validation_data=validation_data,
                       shuffle=shuffle)
