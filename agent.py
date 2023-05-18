import tensorflow as tf

FRAMECOUNT = 4


class Agent():

    def __init__(self, input_image_size):
        self.IMAGE_DIMENSIONS = input_image_size
        self.modelNetwork, self.targetNetwork = self.initializeModels(self.IMAGE_DIMENSIONS[0],
                                                                      self.IMAGE_DIMENSIONS[1])
        self.learningRate = 0.01
        self.actionCount = 3
        self.epsilon = 0.75

    def initializeModels(self, inputWidth, inputHeight):
        modelNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(FRAMECOUNT, inputHeight, inputWidth)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(3, activation="linear")
        ])

        targetNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(FRAMECOUNT, inputHeight, inputWidth)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(3, activation="linear")
        ])

        targetNetwork.set_weights(self.modelNetwork.get_weights())

        modelNetwork.compile(optimizer='adam',
                             loss="mean_squared_error",
                             metrics=["accuracy"])
        targetNetwork.compile(optimizer='adam',
                              loss="mean_squared_error",
                              metrics=["accuracy"])

        return modelNetwork, targetNetwork

    def takeAction(self, state) -> float:
        output = self.modelNetwork(state, training=False)
        print(output)
        actionToTake = tf.math.argmax(output, axis=1)
        print(actionToTake)
        return actionToTake
