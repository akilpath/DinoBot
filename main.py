import sys
import matplotlib.pyplot as plot
import pyautogui
import pyglet.window
import numpy as np
from PIL import Image
import time
import tracemalloc
from timer import Timer
from agent import Agent

from game import Game
import psutil


FRAME_BOUNDING_BOX = (1050, 340, 2410, 650)
GAME_DONE_BOX = (1696, 578, 1760, 624)
INPUT_IMAGE_SIZE = (400, 400)
GAME_OVER_STATE = Image.open("./data/gameover.png").convert("L")
GAME_OVER_STATE = np.array(GAME_OVER_STATE)
EPISODE_COUNT = 1000
COPY_COUNT = 30
pyautogui.PAUSE = 0.001
fig, ax = plot.subplots()
x = []
y = []


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
    frame = pyautogui.screenshot()
    gameOverRegion = frame.crop(GAME_DONE_BOX).convert("L")
    gameOverRegion.show()
    return
    agent = Agent(conv = True, frameDim=INPUT_IMAGE_SIZE[0], frameCount=4)
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
                state = np.zeros(shape=(1,4,INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
                for i in range(4):
                    time.sleep(0.01)
                    frame = pyautogui.screenshot()

                    #test to see if the game ended
                    gameOverRegion = np.array(frame.crop(GAME_DONE_BOX).convert("L"))
                    if (gameOverRegion == GAME_OVER_STATE).all():
                        playing = False

                    img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
                    npimg = np.array(img)
                    state[0, i] = npimg
                state = np.transpose(state, (0, 2, 3, 1))

            action = agent.chooseAction(state)

            if playing or action == 2:
                doAction(action)

            nextState = np.zeros(shape=(1, 4, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
            for i in range(4):
                time.sleep(0.01)
                frame = pyautogui.screenshot()

                # test to see if the game ended
                gameOverRegion = np.array(frame.crop(GAME_DONE_BOX).convert("L"))
                if (gameOverRegion == GAME_OVER_STATE).all():
                    playing = False

                img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
                npimg = np.array(img)
                nextState[0, i] = npimg

            nextState = np.transpose(nextState, (0, 2, 3, 1))
            if playing:
                reward = 2
                if action == 0 or action == 1:
                    #bias towards jumping or ducking to allow it to learn more about jumping
                    for i in range(5):
                        agent.saveTempExperience(state, action, reward, nextState)
            else:
                reward = -100
                for i in range(5):
                    agent.saveTempExperience(state, action, reward, nextState)

            agent.saveTempExperience(state, action, reward, nextState)
            state = nextState

        lasted = timer.getElapsed()
        print(f"Time survived: {lasted}")
        x.append(episode)
        y.append(lasted)
        agent.copyExperience()
        agent.train()
        agent.decayEpsilon()
        if episode % COPY_COUNT == 0:
            print("Weights copied")
            agent.copyWeights()
        time.sleep(0.25)
    ax.plot(x, y)
    plot.savefig("./figures/test2.png")

def test():
    game = Game()
    game.run()


# try:
#     main()
# except pyautogui.FailSafeException as e:
#     ax.plot(x, y)
#     plot.savefig("./figures/test2.png")
main()