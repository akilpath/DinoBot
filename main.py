import pyautogui
import tensorflow as tf
import numpy as np
import PIL
import time
from agent import Agent
#(680,710) to (2025,1020)

FRAME_BOUNDING_BOX = (680, 710, 2025, 1020)
INPUT_IMAGE_SIZE = ((2025-680) // 4, (1020 - 710) // 4)

EPISODE_COUNT = 500

def main():
    time.sleep(3)
    agent = Agent(INPUT_IMAGE_SIZE)
    while(True):
        state = np.zeros(shape=(1,4,INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
        for i in range(4):
            time.sleep(0.010)
            frame = pyautogui.screenshot()
            img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
            npimg = np.array(img)
            state[0, i] = npimg
        action = agent.takeAction(state)
        if action == 0:
            pyautogui.press("space")
        elif action == 1:
            pyautogui.press("down")
        time.sleep(0.1)

        reward = 0

        nextState = np.zeros(shape=(1, 4, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
        for i in range(4):
            time.sleep(0.010)
            frame = pyautogui.screenshot()
            img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
            npimg = np.array(img)
            nextState[0, i] = npimg
        pyautogui.keyUp("up")
        pyautogui.keyUp("down")

        agent.saveExperience(state, action, reward, nextState)
        print("forward pass done")

main()