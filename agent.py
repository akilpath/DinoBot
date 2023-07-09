from collections import deque

import tensorflow as tf
import numpy as np
import random
import psutil
#from keras_visualizer import visualizer
import tensorflow.python.framework.errors_impl


class Agent:
    def __init__(self, stateShape, actionSize):
        self.ACTIONSIZE = actionSize
        self.STATESHAPE = stateShape

        self.modelNetwork, self.targetNetwork = self.initializeConvModels(self.STATESHAPE)
        try:
            self.modelNetwork.load_weights("data/weights.h5")
        except Exception as e:
            print(e)

        self.copyWeights()
        self.gamma = 0.9
        self.epsilon = 0.3
        self.decayRate = 0.90
        self.batchSize = 64
        self.epsilonMin = 0.0001

        self.memory = deque(maxlen=100000)
        self.tempExperience = deque(maxlen=450)

    def decayEpsilon(self):
        if self.epsilon <= self.epsilonMin:
            return
        self.epsilon *= self.decayRate

    def saveTempExperience(self, state, action, reward, nextState):
        self.tempExperience.appendleft((state, action, reward, nextState))

    def copyExperience(self):
        self.memory += self.tempExperience
        self.tempExperience.clear()

    def copyWeights(self):
        self.targetNetwork.set_weights(self.modelNetwork.get_weights())

    def train(self):
        if len(self.memory) < self.batchSize:
            print("Not enough exp to train")
            return

        batch = random.sample(self.memory, self.batchSize)

        states = []
        actions = []
        rewards = []
        nextStates = []
        for i in range(self.batchSize):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            nextStates.append(batch[i][3])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        nextStates = np.array(nextStates)

        predictedQ = self.modelNetwork.predict(states, verbose=0)
        targetQ = self.modelNetwork.predict(nextStates, verbose=0)
        for i in range(self.batchSize):
            if rewards[i] == -10:
                predictedQ[i, actions[i]] = rewards[i]
            else:
                predictedQ[i, actions[i]] = rewards[i] + self.gamma*np.max(targetQ[i])
        self.modelNetwork.fit(states, predictedQ, verbose=0)
        # for state, action, reward, nextState in batch:
        #     predictedQ = self.modelNetwork.predict(state, verbose=0)
        #
        #     targetQ = self.targetNetwork.predict(nextState, verbose=0)
        #     if reward == -10:
        #         predictedQ[0, action] = reward
        #     else:
        #         predictedQ[0, action] = reward + self.gamma * np.max(targetQ, axis=1)
        #     self.modelNetwork.fit(state, predictedQ, verbose=0)
        print("Finished Training")
        print(f"Memory length: {len(self.memory)}")
        print(f"Memory Usage: {psutil.virtual_memory()[3] / float((pow(10, 9)))}")

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

        return modelNetwork, targetNetwork

    def initializeConvModels(self, inputShape):
        def buildModel():
            return tf.keras.models.Sequential([
                tf.keras.layers.Rescaling(1. / 255, input_shape=inputShape),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation="leaky_relu"),
                tf.keras.layers.Dense(128, activation="leaky_relu"),
                tf.keras.layers.Dense(3, activation="linear")
            ])
        model = buildModel()

        target = buildModel()

        model.compile(optimizer='adam',
                      loss="huber",
                      metrics=["accuracy"])
        target.compile(optimizer='adam',
                       loss="huber",
                       metrics=["accuracy"])

        return model, target

    def chooseAction(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(3)
        output = self.modelNetwork.predict(state[np.newaxis, :], verbose=0)
        actionToTake = np.argmax(output, axis=1)
        return actionToTake[0]

    def saveModel(self):
        tf.saved_model.save(self.modelNetwork, "./")

    def saveWeights(self):
        self.modelNetwork.save_weights("data/weights.h5", save_format="h5")
