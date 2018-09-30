#!/usr/bin/env python3
import time
import pyautogui as gui


def launch_dota():
    # bring up spotlight search
    gui.hotkey('command', 'space')
    time.sleep(1)

    # search for steam (assumes it is already launched)
    gui.typewrite('steam', interval=0.1)
    gui.press('enter')
    time.sleep(1)

    gui.click(x=335, y=225, pause=20)


def restart_game():
    gui.press('\\', pause=0.1)
    gui.typewrite('restart', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.press('\\', pause=0.1)


def start_game():
    # start
    gui.click(x=1200, y=850, pause=0.5)
    # create lobby
    gui.click(x=1170, y=275, pause=2)
    # join coaches
    gui.click(x=1100, y=365, pause=2)
    # start game
    gui.click(x=1250, y=850, pause=1)


def set_timescale():
    gui.press('\\', pause=0.1)
    gui.typewrite('sv_cheats 1', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.typewrite('host_timescale 4', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.press('\\', pause=0.5)


def run():
    launch_dota()
    set_timescale()
    start_game()


if __name__ == '__main__':
    run()