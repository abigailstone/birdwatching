#!/usr/bin/env python3
"""
Simple camera to make testing easier
Illuminates button and takes picture on button press, saves to Pictures directory
"""
import time
import sys
import contextlib
import subprocess

from picamera import PiCamera

from aiy.board import Board, Led
from aiy.leds import Color, Leds, Pattern, PrivacyLed

savedir = "~/Pictures/manual/"


def take_photo():
    """
    uses raspistill to take a photo
    """

    timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')

    cmd = "raspistill -w 1640 -h 922 -o "+savedir+timestamp+".jpg -roi 0.33,0.45,0.33,0.33"
    subprocess.call(cmd, shell=True)

    print("Saved photo at "+savedir+timestamp)



def main():

    print("Simple camera running")

    with contextlib.ExitStack() as stack:
        leds = stack.enter_context(Leds())
        board = stack.enter_context(Board())


        while True:
            leds.update(Leds.rgb_on(Color.GREEN))
            board.button.wait_for_press()
            leds.update(Leds.rgb_on(Color.RED))
            board.button.wait_for_release()
            take_photo()



if __name__ == '__main__':
    main()
