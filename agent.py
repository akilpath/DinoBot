from collections import deque

import tensorflow as tf
import numpy as np
import random


class Agent:
    def __init__(self):
        self.modelNetwork, self.targetNetwork = self.initializeModels()
        self.gamma = 0.2
        self.epsilon = 0.8
        self.decayRate = 0.999
        self.actionCount = 3
        self.batchSize = 32
        self.memory = deque(maxlen = 3000)

    def decayEpsilon(self):
        self.epsilon *= self.decayRate

    def saveExperience(self, state, action, reward, nextState):
        self.memory.append((state, action, reward, nextState))
        pass

    def copyWeights(self):
        self.targetNetwork.set_weights(self.modelNetwork.get_weights())

    def train(self):
        if self.batchSize > len(self.memory):
            batch = self.memory
        else:
            batch = random.sample(self.memory,self.batchSize)

        for state, action, reward, nextState in batch:
            predictedQ = self.modelNetwork.predict(state, verbose=0)

            targetQ = self.targetNetwork.predict(nextState, verbose=0)
            predictedQ[0, action] = reward + self.gamma*np.amax(targetQ) - predictedQ[0, action]
            self.modelNetwork.fit(state, predictedQ, verbose=0)


    def initializeModels(self):
        modelNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(128, activation= "relu"),
            tf.keras.layers.Dense(3, activation="linear")
        ])
        modelNetwork.summary()
        targetNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(3, activation="linear")
        ])

        targetNetwork.set_weights(modelNetwork.get_weights())

        modelNetwork.compile(optimizer='adam',
                             loss="mse",
                             metrics=["accuracy"])
        targetNetwork.compile(optimizer='adam',
                              loss="mse",
                              metrics=["accuracy"])

        return modelNetwork, targetNetwork

    def chooseAction(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(3)

        output = self.modelNetwork(state, training=False)
        actionToTake = tf.math.argmax(output, axis=1)
        return actionToTake
