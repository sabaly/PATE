import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from data_loader import *
from partition import *
import numpy as np

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

def train_teacher(x_train, y_train, nb_epochs=20):
    model = st_model(x_train.shape[1:])

    model.fit(x_train, y_train, epochs = nb_epochs, versbose=False)

    return model

def train_teachers(dataset, nb_tchrs, nb_epochs=20):
    x_train, x_test, y_train, y_test = get(dataset)
    subsets = basic_partition(x_train, y_train, nb_tchrs)

    fi = []
    print("\nTraining teachers...", end="")
    
    for subset in subsets:
        x_train, y_train = subset
        x_train, y_train = np.array(x_train), np.array(y_train)

        model = define_model((x_train.shape[-1],))

        model.fit(x_train, y_train, epochs=nb_epochs, verbose=True)
        fi.append(model)
    print("Done")

    return fi, (x_test, y_test)


def eval_teacher_model(model, x_test, y_test):
    print('Evaluation of a teacher model')
    eval1 = model.evaluate(x_test, y_test)
    print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
    


