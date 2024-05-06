import tensorflow as tf
from data_loader import *
import numpy as np
from aggregator import *


def define_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(128,input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    
    return model

def train_student(x_train, labelizer, nb_epochs=20):
    y_train = labelizer(x_train)

    model = st_model(x_train.shape[1:])

    model.fit(x_train, y_train, epochs = nb_epochs, versbose=False)

    return model, (x_test, y_test, true_y_test)

def eval_student_model(model, x_test, true_y_test, labelizer):
    y_test = labelizer(x_test)
    print('Test 2 : evaluation the student on aggregated labels')
    eval1 = model.evaluate(x_test, y_test)
    print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
    print('Test 2 : evaluation the student on true labels')
    eval2 = model.evaluate(x_test, true_y_test)
    print(f"**** Results \n\t-loss : {eval2[0]}\n\t-accuracy : {eval2[1]}")
    


