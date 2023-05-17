import pyautogui
import tensorflow as tf
import numpy as np
import PIL
import time
#(680,710) to (2025,1020)


def main():
    print(pyautogui.KEY_NAMES)
    time.sleep(3)
    frame = pyautogui.screenshot()
    img = frame.crop((680, 710, 2025, 1020))
    img.show()
    input_image_size = (img.size[0]//4, img.size[1]//4)
    img = img.resize((input_image_size[0], input_image_size[1]))
    img = img.convert("L")
    img.show()
    while(True):
        time.sleep(2)
        pyautogui.keyDown("down")

main()