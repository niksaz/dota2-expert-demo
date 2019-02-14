import time
import subprocess
import pyautogui as gui


RIGHT_BOT_BUTTON_X = 906
RIGHT_BOT_BUTTON_Y = 809

INTERVAL = 0.01
DURATION = 0.5
PAUSE = 1

restart_episode_called = 0
RESTART_AFTER_EPISODES = 100


def prepare_steam_client():
    if _is_steam_launched():
        _focus_steam_window()
    else:
        # "Search your computer" Ubuntu function
        gui.click(x=32, y=56)
        gui.typewrite('steam', interval=INTERVAL, pause=PAUSE)

        # Run the first option
        gui.press('enter', pause=PAUSE)
        time.sleep(30)


def prepare_dota_client():
    if _is_dota_launched():
        _focus_dota_window()
    else:
        # Click on the library option
        gui.click(x=307, y=85, pause=PAUSE)
        # Search for Dota 2 in the library
        gui.click(x=150, y=120)
        gui.typewrite('dota', interval=INTERVAL)

        # Press play
        gui.click(x=416, y=236, pause=30)
        calibrate_dota_client()
        start_game()


def start_game():
    # Leave the game if there is one
    gui.click(x=832, y=786, duration=DURATION, pause=PAUSE)
    # Confirm it
    gui.click(x=580, y=571, pause=4*PAUSE)

    # Start
    gui.click(x=RIGHT_BOT_BUTTON_X, y=RIGHT_BOT_BUTTON_Y, duration=DURATION, pause=PAUSE)
    # Create lobby
    gui.click(x=854, y=424, duration=DURATION, pause=PAUSE)
    # Join coaches
    gui.click(x=807, y=484, duration=DURATION, pause=PAUSE)
    # Start game
    gui.click(x=RIGHT_BOT_BUTTON_X, y=RIGHT_BOT_BUTTON_Y, duration=DURATION, pause=PAUSE)


def restart_game():
    global restart_episode_called
    restart_episode_called += 1
    if restart_episode_called >= RESTART_AFTER_EPISODES:
        restart_episode_called = 0
        close_dota_client()
        close_steam_client()

    # Add a full restart here after a certain number of episodes
    prepare_steam_client()
    prepare_dota_client()

    # Enter the restart command
    gui.press('\\', pause=PAUSE)
    gui.typewrite('restart', interval=INTERVAL)
    gui.press('enter', pause=PAUSE)
    gui.press('\\', pause=PAUSE)

    # Wait until the in-game UI is visible
    cnt = 0
    while True:
        point = gui.locateOnScreen('images/inactive_arrow.png')
        if point:
            break
        cnt += 1
        if cnt >= 120:
            restart_game()

    # Start the game right away
    gui.press('\\', pause=PAUSE)
    gui.typewrite('dota_start_game', interval=INTERVAL)
    gui.press('enter')
    gui.press('\\', pause=PAUSE)


def close_steam_client():
    if not _is_steam_launched():
        return
    _focus_steam_window()

    # Click on the steam menu option
    gui.click(x=101, y=51, pause=PAUSE)
    # Pick the exit option
    gui.click(x=101, y=199, pause=PAUSE)
    time.sleep(30)


def close_dota_client():
    if not _is_dota_launched():
        return
    _focus_dota_window()

    # Bring up the menu
    gui.click(x=256, y=256, pause=2*PAUSE)
    # Disconnect
    gui.click(x=RIGHT_BOT_BUTTON_X, y=RIGHT_BOT_BUTTON_Y, pause=2*PAUSE)
    # Confirm it
    gui.click(x=585, y=585, pause=4*PAUSE)
    # Exit
    gui.click(x=1022, y=256, pause=2*PAUSE)
    # Confirm it and wait for complete closure
    gui.click(x=580, y=568, pause=PAUSE)
    time.sleep(30)


def calibrate_dota_client():
    gui.press('\\', pause=PAUSE)
    gui.typewrite('sv_cheats 1', interval=INTERVAL)
    gui.press('enter', pause=PAUSE)
    gui.typewrite('host_timescale 5', interval=INTERVAL)
    gui.press('enter', pause=PAUSE)
    gui.press('\\', pause=PAUSE)


def _is_steam_launched():
    return _run_cmd('ps -ef | grep steam').find('steam.sh') != -1


def _is_dota_launched():
    return _run_cmd('ps -ef | grep dota').find('dota 2 beta') != -1


def _focus_steam_window():
    # wmctrl detects the Steam's window as N/A.
    windows = _run_cmd('wmctrl -l')
    for window_info in windows.splitlines():
        if window_info.find('N/A') != -1:
            window_id = window_info[:10]
            _run_cmd('wmctrl -i -a ' + window_id)
    time.sleep(DURATION)


def _focus_dota_window():
    _run_cmd('wmctrl -a "Dota 2"')
    time.sleep(DURATION)


def _run_cmd(cmd):
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return output.decode('utf-8')
