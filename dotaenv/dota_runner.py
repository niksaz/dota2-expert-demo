#!/usr/bin/env python3
import time
import pyautogui as gui


def launch_dota():
    # bring up spotlight search
    gui.hotkey('command', 'space')
    time.sleep(1)

    # search for steam (assuming it is already launched)
    gui.typewrite('steam', interval=0.1)
    gui.press('enter')
    time.sleep(1)

    # search for Dota 2 in the library
    gui.click(x=50, y=110)
    gui.typewrite('dota', interval=0.1)

    # press play
    gui.click(x=335, y=225, pause=20)


def restart_game():
    gui.press('\\', pause=0.1)
    gui.typewrite('restart', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.press('\\', pause=0.1)


def close_game():
    # bring up the menu
    gui.click(x=373, y=223, pause=0.5)
    # disconnect
    gui.click(x=980, y=671, pause=0.5)
    # confirm it
    gui.click(x=680, y=488, pause=2)
    # exit
    gui.click(x=1068, y=228, pause=0.5)
    # confirm it
    gui.click(x=680, y=475, pause=5)


def start_game():
    # start
    gui.click(x=974, y=668, pause=0.5)
    # create lobby
    gui.click(x=892, y=353, pause=2)
    # join coaches
    gui.click(x=899, y=408, pause=2)
    # start game
    gui.click(x=974, y=668, pause=1)


def set_timescale():
    gui.press('\\', pause=0.1)
    gui.typewrite('sv_cheats 1', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.typewrite('host_timescale 5', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.press('\\', pause=0.5)


def run():
    launch_dota()
    set_timescale()
    start_game()


if __name__ == '__main__':
    run()
