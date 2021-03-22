import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = tf.keras.utils.normalize(x, axis=1)

dense_layers = [0]
dense_layer_sizes = [512]
conv_layers = [5]
conv_layer_sizes = [32]

for dense_layer in dense_layers:
    for dense_layer_size in dense_layer_sizes:
        for conv_layer in conv_layers:
            for conv_layer_size in conv_layer_sizes:
                NAME = "dense-{}-size-{}-conv-{}-size-{}-{}".format(dense_layer, dense_layer_size, conv_layer, conv_layer_size, int(time.time()))
                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model = Sequential()

                for i in range(conv_layer):
                    model.add(Conv2D(conv_layer_size, (3, 3), input_shape=x.shape[1:], activation=tf.nn.relu))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                for i in range(dense_layer):
                    model.add(Dense(dense_layer_size, activation=tf.nn.relu))

                model.add(Dense(1, activation=tf.nn.sigmoid))

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=['accuracy'])

                model.fit(x, y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])
                model.summary()

model.save("x64-cnn-1.model")