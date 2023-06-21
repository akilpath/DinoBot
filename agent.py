from collections import deque

import tensorflow as tf
import numpy as np
import random
import psutil
from keras_visualizer import visualizer


class Agent:
    def __init__(self, stateSize, actionSize):
        self.ACTIONSIZE = actionSize
        self.STATESIZE = stateSize

        self.modelNetwork, self.targetNetwork = self.initializeModels()
        self.copyWeights()
        self.gamma = 0.95
        self.epsilon = 0.5
        self.decayRate = 0.90
        self.batchSize = 64
        self.epsilonMin = 0.0001
        self.episodeCount = 0

        self.memory = deque(maxlen=50000)
        self.tempExperience = deque(maxlen=450)

    def decayEpsilon(self):
        if self.epsilon <= self.epsilonMin:
            return
        self.epsilon *= self.decayRate

    def saveTempExperience(self, state, action, reward, nextState):
        self.tempExperience.appendleft((state, action, reward, nextState))

    def copyExperience(self):
        self.memory += self.tempExperience

    def copyWeights(self):
        self.targetNetwork.set_weights(self.modelNetwork.get_weights())

    def train(self):
        if len(self.memory) < self.batchSize:
            return

        batch = random.sample(self.memory, self.batchSize)

        for state, action, reward, nextState in batch:
            predictedQ = self.modelNetwork.predict(state, verbose=0)

            targetQ = self.targetNetwork.predict(nextState, verbose=0)
            if reward == -10:
                predictedQ[0, action] = reward
            else:
                predictedQ[0, action] = reward + self.gamma * np.max(targetQ, axis=1)
            self.modelNetwork.fit(state, predictedQ, verbose=0)
        print("Finished Training")
        print(f"Memory length: {len(self.memory)}")
        print(f"Computer Memory Usage: {psutil.virtual_memory()[3] / float((pow(10, 9)))}")

    def initializeModels(self):
        modelNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.STATESIZE),
            tf.keras.layers.Dense(32, activation="leaky_relu"),
            tf.keras.layers.Dense(8, activation="leaky_relu"),
            tf.keras.layers.Dense(self.ACTIONSIZE, activation="linear")
        ])
        modelNetwork.summary()
        targetNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.STATESIZE),
            tf.keras.layers.Dense(32, activation="leaky_relu"),
            tf.keras.layers.Dense(8, activation="leaky_relu"),
            tf.keras.layers.Dense(self.ACTIONSIZE, activation="linear")
        ])

        modelNetwork.compile(optimizer='adam',
                             loss="huber",
                             metrics=["accuracy"])
        targetNetwork.compile(optimizer='adam',
                              loss="huber",
                              metrics=["accuracy"])

        #visualizer(modelNetwork, file_name="visualization", file_format="png", view=True)

        return modelNetwork, targetNetwork

    def chooseAction(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(3)

        output = self.modelNetwork.predict(state, verbose=0)
        actionToTake = np.argmax(output, axis=1)
        return actionToTake[0]

    def saveModel(self):
        tf.saved_model.save(self.modelNetwork, "./")
