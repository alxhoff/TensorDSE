import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0



model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28,28,1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=12,kernel_size=(3, 3), activation=LeakyReLU()),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])


model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

model.save('saved_test_model')