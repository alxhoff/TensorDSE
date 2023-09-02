def GetData():
    import tensorflow as tf

    (
        (images_train, labels_train),
        (images_test, labels_test),
    ) = tf.keras.datasets.mnist.load_data()

    images_train = images_train.reshape(
        images_train.shape[0], images_train.shape[1], images_train.shape[2], 1
    )
    images_test = images_test.reshape(
        images_test.shape[0], images_test.shape[1], images_test.shape[2], 1
    )
    input_tensor_shape = (images_train.shape[1], images_train.shape[2], 1)
    images_train = images_train.astype("float32")
    images_test = images_test.astype("float32")
    images_train /= 255
    images_test /= 255

    return {
        "train_data": images_train,
        "test_data": images_test,
        "train_labels": labels_train,
        "test_labels": labels_test,
        "input_tensor_shape": input_tensor_shape,
    }


def GetInputShape():

    return (28, 28, 1)