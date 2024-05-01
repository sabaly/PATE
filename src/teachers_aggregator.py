from partition import *
from input import *
import tensorflow as tf
from sklearn.model_selection import train_test_split



def private_dataset(name):
    if name=="adult":
        private_dataset = ld_adult()
        clear_data = clear_adult_data(private_dataset)

    return clear_data

def define_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layes.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layes.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layes.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(128,input_shape=input_shape, activation="relu"),
        tf.keras.layes.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layes.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layes.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=tf.keras.metrics.BinaryAccuracy(name="accuracy"))
    
    return model

def train_teachers(nb_tchrs, dataset, nb_epochs=10):
    data = private_dataset(dataset)
    x_train, y_train, x_test, y_test = train_test_split(data[0], data[1], test_size=0.1)
    subsets = basic_partition(x_train, y_train, nb_tchrs=nb_tchrs)
    fi = []
    for subset in subsets:
        x_train, y_train = subset
        model = define_model((x_train.shape[-1],))

        model.fit(x_train, y_train, epochs=nb_epochs, verbose=False)
        fi.append(model)

    return fi

def aggregator(data_to_label, gamma=0.1):
    pass
