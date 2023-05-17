import tensorflow as tf
FRAMECOUNT = 4
class Agent():

    def __init__(self, input_image_size):
        inputWidth, inputHeight = input_image_size
        self.modelNetwork = tf.keras.models.Sequential(
            tf.keras.layers.Rescaling(1. / 255, input_shape=(FRAMECOUNT, inputWidth, inputHeight)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation="linear")
        )

        self.targetNetwork = tf.keras.models.Sequential(
            tf.keras.layers.Rescaling(1. / 255, input_shape=(FRAMECOUNT, inputWidth, inputHeight)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation="linear")
        )

        self.targetNetwork.set_weights(self.modelNetwork.get_weights())
