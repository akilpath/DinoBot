import pyautogui
import tensorflow as tf
import numpy as np
import PIL
import time
from agent import Agent
#(680,710) to (2025,1020)

FRAME_BOUNDING_BOX = (680, 710, 2025, 1020)
INPUT_IMAGE_SIZE = ((2025-680) // 4, (1020 - 710) // 4)

def main():
    time.sleep(3)
    frame = pyautogui.screenshot()
    img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
    npimg = np.array(img)
    print(f"Dimensions: {npimg.shape}")
    agent = Agent(INPUT_IMAGE_SIZE)
    while(True):
        frames = np.zeros(shape=(4,INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
        for i in range(4):
            time.sleep(0.010)
            frame = pyautogui.screenshot()
            frame = pyautogui.screenshot()
            img = frame.crop(FRAME_BOUNDING_BOX).resize(INPUT_IMAGE_SIZE).convert("L")
            npimg = np.array(img)
            frames[i] = npimg
        frames = frames[np.newaxis, :]
        action = agent.takeAction(frames)
        if action == 0:
            pyautogui.keyDown("space")
        elif action == 1:
            pyautogui.keyDown("down")
        time.sleep(0.25)
        pyautogui.keyUp("up")
        pyautogui.keyUp("down")
        print("forward pass done")

main()