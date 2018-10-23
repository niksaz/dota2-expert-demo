import time
import subprocess
import pyautogui as gui


def make_sure_dota_is_launched():
    if _is_dota_launched():
        _bring_into_focus()
        return

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


def restart_game():
    _bring_into_focus()

    gui.press('\\', pause=0.1)
    gui.typewrite('restart', interval=0.05, pause=0.3)
    gui.press('enter', pause=0.1)
    gui.press('\\', pause=0.1)
    time.sleep(10)
    # Press keys to speed up Dota 2 launching
    gui.press('esc', pause=1)
    gui.press('esc', pause=1)
    gui.press('esc', pause=1)
    gui.press('esc', pause=1)
    gui.press('esc', pause=1)


def close_game():
    _bring_into_focus()

    # bring up the menu
    gui.click(x=373, y=223, pause=1)
    # disconnect
    gui.click(x=980, y=671, pause=1)
    # confirm it
    gui.click(x=680, y=488, pause=2)
    # exit
    gui.click(x=1068, y=228, pause=1)
    # confirm it and wait for complete closure
    gui.click(x=680, y=475, pause=15)


def _bring_into_focus():
    gui.moveTo(967, 1000, pause=0.8)
    gui.click(967, 1000, pause=0.1)
    gui.click(750, 400, pause=0.1)


def _is_dota_launched():
    return _find_process("dota").find(b"dota 2 beta") != -1


def _find_process(process_name):
    ps = subprocess.Popen("ps -ef | grep " + process_name,
                          shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return output


def run():
    make_sure_dota_is_launched()
    set_timescale()
    start_game()


if __name__ == '__main__':
    run()
