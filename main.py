import matplotlib.pyplot as plot
import pyautogui
import numpy as np
from PIL import Image
from time import sleep
from timer import Timer
from agent import Agent

FRAME_BOUNDING_BOX = (1120, 470, 2340, 730)
GAME_DONE_BOX = (1696, 578, 1760, 624)
INPUT_IMAGE_SIZE = (200, 200)
GAME_OVER_STATE = np.array(Image.open("./data/gameover.png").convert("L"))
EPISODE_COUNT = 200
COPY_COUNT = 30
pyautogui.PAUSE = 0.001
STATE_SHAPE = (INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3)
fig, ax = plot.subplots()
x = []
y = []
TEST_NAME = "test11-2"


def doAction(action=2):
    pyautogui.keyUp("space")
    pyautogui.keyUp("down")

    if action == 0:
        pyautogui.keyDown("space")
    elif action == 1:
        pyautogui.keyDown("down")


def main():
    print("Starting")
    agent = Agent(STATE_SHAPE, 3)
    sleep(4)
    timer = Timer()
    for episode in range(EPISODE_COUNT):
        print(f"Starting Episode {episode}")
        print(f"Agent epsilon: {agent.epsilon}")
        pyautogui.press("space")
        timer.startTimer()
        playing = True
        lastState = None
        lastAction = -1
        # delay of 1.5 seconds prevents a.i from logging information at the beginning of the game.
        sleep(1.5)
        while playing:
            if lastState is None:
                imgs = []
                for i in range(3):
                    sleep(0.01)
                    frame = pyautogui.screenshot().convert("L")

                    img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE)
                    imgs.append(np.array(img))
                lastState = np.array(imgs)
                lastState = np.transpose(lastState, (1, 2, 0))
            else:
                imgs = []
                for i in range(3):
                    sleep(0.01)
                    frame = pyautogui.screenshot().convert("L")

                    if (np.array(frame.crop(GAME_DONE_BOX)) == GAME_OVER_STATE).all():
                        playing = False

                    img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE)
                    imgs.append(np.array(img))
                state = np.array(imgs)
                state = np.transpose(state, (1, 2, 0))
                if not playing:
                    for i in range(5):
                        agent.saveTempExperience(lastState, lastAction, -10, state)
                else:
                    if lastAction != 2:
                        agent.saveTempExperience(lastState, lastAction, 2, state)
                    else:
                        agent.saveTempExperience(lastState, lastAction, 1, state)
                lastState = state

            lastAction = agent.chooseAction(lastState)
            if playing:
                doAction(lastAction)
        #reset keys
        doAction()
        lasted = timer.getElapsed()
        print(f"Time survived: {lasted}")
        x.append(episode)
        y.append(lasted)
        agent.copyExperience()
        agent.train()
        print("Finished training")
        agent.decayEpsilon()
        if episode % COPY_COUNT == 0:
            agent.copyWeights()
        sleep(0.25)
    agent.saveWeights()
    print("Weights Saved")
    ax.plot(x, y)
    plot.savefig(f"./figures/{TEST_NAME}.png")


def test():
    print("Starting")
    sleep(4)
    images = []
    imgs = []
    for i in range(3):
        sleep(0.01)
        frame = pyautogui.screenshot().convert("L")

        img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE)
        images.append(img)
        imgs.append(np.array(img))
    lastState = np.array(imgs)
    lastState = np.transpose(lastState, (1, 2, 0))[np.newaxis, :]
    for image in images:
        image.show()



main()
