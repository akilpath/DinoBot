import sys
import matplotlib.pyplot as plot
import pyautogui
import pyglet.window
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from timer import Timer
from agent import Agent
from game import Game

EPISODE_COUNT = 2000
COPY_COUNT = 30
pyautogui.PAUSE = 0.001
fig, ax = plot.subplots()


def doAction(action):
    pyautogui.keyUp("space")
    pyautogui.keyUp("down")

    if action == 0:
        pyautogui.keyDown("space")
    elif action == 1:
        pyautogui.keyDown("down")


def main():
    print("Starting")
    time.sleep(4)
    agent = Agent()
    stepCount = 0

    timer = Timer()

    for episode in range(EPISODE_COUNT):
        print(f"Episode {episode}")
        print(f"Agent epsilon: {agent.epsilon}")
        pyautogui.press("space")
        timer.startTimer()
        playing = True
        state = None
        # delay of 2 seconds prevents a.i from logging information at the beginning of the game.
        time.sleep(2)
        while playing:
            stepCount += 1
            if state is None:
                state = np.transpose(state, (0, 2, 3, 1))

            action = agent.chooseAction(state)

            if playing or action == 2:
                doAction(action)


            nextState = np.transpose(nextState, (0, 2, 3, 1))
            if playing:
                reward = 2
                if action == 0 or action == 1:
                    #bias towards jumping or ducking to allow it to learn more about jumping
                    for i in range(5):
                        agent.saveExperience(state, action, reward, nextState)
            else:
                reward = -100
                for i in range(5):
                    agent.saveExperience(state, action, reward, nextState)

            agent.saveExperience(state, action, reward, nextState)
            state = nextState

        lasted = timer.getElapsed()
        print(f"Time survived: {lasted}")
        # x[episode-1] = episode
        # y[episode-1] = lasted
        agent.train()
        agent.decayEpsilon()
        if episode % COPY_COUNT == 0:
            print("Weights copied")
            agent.copyWeights()
        time.sleep(0.25)
    #ax.plot(x, y)
    #plot.savefig("./figures/test2.png")


def test():
    x = []
    y = []
    game = Game()
    game.run()
    ax.plot(x, y)
    # window = pyglet.window.Window()
    # label = pyglet.text.Label('Hello, world',
    #                           font_name='Times New Roman',
    #                           font_size=36,
    #                           x=window.width // 2, y=window.height // 2,
    #                           anchor_x='center', anchor_y='center')
    # @window.event
    # def on_draw():
    #     window.clear()
    #     label.draw()
    #pyglet.app.run()


# try:
#     main()
# except pyautogui.FailSafeException as e:
#     ax.plot(x, y)
#     plot.savefig("./figures/test2.png")
test()
