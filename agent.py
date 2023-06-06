from collections import deque

import tensorflow as tf
import numpy as np
import random


class Agent:
    def __init__(self):
        self.modelNetwork, self.targetNetwork = self.initializeModels()
        self.gamma = 0.9
        self.epsilon = 0.3
        self.decayRate = 0.90
        self.batchSize = 32
        self.epsilonMin = 0.0001
        self.episodeCount = 0

        self.memory = deque(maxlen=4000)

    def decayEpsilon(self):
        if self.epsilon <= self.epsilonMin:
            return
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

    def initializeModels(self):
        modelNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_dim=9),
            tf.keras.layers.Dense(64, activation="leaky_relu"),
            tf.keras.layers.Dense(32, activation="leaky_relu"),
            tf.keras.layers.Dense(3, activation="linear")
        ])
        modelNetwork.summary()
        targetNetwork = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_dim=9),
            tf.keras.layers.Dense(64, activation="leaky_relu"),
            tf.keras.layers.Dense(32, activation="leaky_relu"),
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

        output = self.modelNetwork.predict(state,verbose=0)
        actionToTake = np.argmax(output,axis=1)
        return actionToTake[0]

    def saveModel(self):
        tf.saved_model.save(self.modelNetwork, "./")
