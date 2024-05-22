import tensorflow as tf
from data_loader import *
import numpy as np
from aggregator import *


def define_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(input_shape),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Recall(name="recall")])
    
    return model

def train_student(x_train, y_train, nb_epochs=100, verbose=True):
    model = define_model(x_train.shape[1:])
    if verbose:
        print("Training student...", end="")
    model.fit(x_train, y_train, epochs = nb_epochs, verbose=False)
    if verbose:
        print("Done")
    return model

def eval_student_model(model, x_test, true_y_test, y_test, verbose=True):
    if verbose:
        print('Test 1 : evaluation the student on aggregated labels')
    eval1 = model.evaluate(x_test, y_test)
    if verbose:
        print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
        print("-------------------")
        print('Test 2 : evaluation the student on true labels')
    eval2 = model.evaluate(x_test, true_y_test)
    if verbose:
        print(f"**** Results \n\t-loss : {eval2[0]}\n\t-accuracy : {eval2[1]}")
    return y_test
    


