import sys
import matplotlib.pyplot as plot
import pyautogui
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from timer import Timer
from agent import Agent
#(1050,340) to (2410,650)
#Game over location (1696,512) to (1760, 568)

FRAME_BOUNDING_BOX = (1050, 340, 2410, 650)
GAME_DONE_BOX = (1696, 512, 1760, 568)
INPUT_IMAGE_SIZE = ((FRAME_BOUNDING_BOX[2] - FRAME_BOUNDING_BOX[0]) // 5,
                    (FRAME_BOUNDING_BOX[3] - FRAME_BOUNDING_BOX[1]) // 5)
GAME_OVER_STATE = Image.open("./data/gameover.png").convert("L")
GAME_OVER_STATE = np.array(GAME_OVER_STATE)
EPISODE_COUNT = 1000
COPY_COUNT = 30

# def test():
#     print("Start")
#     time.sleep(5)
#     frame = pyautogui.screenshot()
#     cropped = frame.crop(GAME_DONE_BOX).convert("L")
#     npcrop = np.array(cropped)
#     cropped.show()
#     print(f"dim: {npcrop.shape}" )
#     print(f"dim: {GAME_OVER_STATE.shape}")
#     print(f"Eval: {(npcrop == GAME_OVER_STATE).all()}")

def main():
    time.sleep(4)
    agent = Agent(INPUT_IMAGE_SIZE)
    stepCount = 0

    timer = Timer()

    #plotting
    x = np.zeros(dtype=int, shape=EPISODE_COUNT)
    y = np.zeros(dtype=float, shape=EPISODE_COUNT)
    fig, ax = plot.subplots()

    for episode in range(EPISODE_COUNT):
        print(f"Episode {episode}")
        print(f"Agent epsilon: {agent.epsilon}")
        pyautogui.press("space")
        timer.startTimer()
        playing = True
        while playing:
            stepCount += 1
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

            action = agent.chooseAction(state)

            if playing:
                if action == 0:
                    pyautogui.press("space")
                elif action == 1:
                    pyautogui.press("down")
            else:
                action = 2

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

            if playing:
                reward = 1
            else:
                reward = -10

            agent.saveExperience(state, action, reward, nextState)

            if stepCount % COPY_COUNT == 0:
                print("Weights copied")
                agent.copyWeights()
                stepCount = 0
        lasted = timer.getElapsed()
        print(f"Time survived: {lasted}")
        x[episode-1] = episode
        y[episode-1] = lasted
        agent.train()
        agent.decayEpsilon()
        time.sleep(0.25)

    ax.plot(x,y)
    plot.savefig("test1.png")

main()
#test()
