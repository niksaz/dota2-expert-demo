import time
import json
import subprocess
import pyautogui as gui
from pyscreeze import ImageNotFoundException

with open('gui_config.json', 'r') as finput:
    config = json.load(finput)
print('GUI config', config)

STEAM_LIBRARY = config['steam_library_button']
STEAM_SEARCH = config['steam_search_box']
STEAM_PLAY = config['steam_play_button']
STEAM_MENU = config['steam_menu_option']
STEAM_EXIT = config['steam_exit_option']

RIGHT_BOT_BUTTON = config['dota_right_bottom_button']
LEAVE_BUTTON = config['dota_leave_game_button']
MENU_CONFIRM_BUTTON = config['dota_menu_confirm_button']
CREATE_LOBBY_BUTTON = config['dota_create_lobby_button']
JOIN_COACHES_BUTTON = config['dota_join_coaches_button']
INGAME_MENU_BUTTON = config['dota_ingame_menu_button']
INGAME_CONFIRM_BUTTON = config['dota_ingame_confirm_button']
EXIT_BUTTON = config['dota_exit_button']

TYPEWRITE_INT = config['typewrite_interval']
MOUSE_DURATION = config['mouse_movement_duration']
PAUSE = config['in_between_pause']

RESTART_AFTER_EPISODES = config['restart_client_every_nth']

episodes_since_last_restart = 0


def prepare_steam_client():

    while True:
        if _is_steam_launched():
            _focus_steam_window()
            break
        else:
            gui.press('winleft', pause=PAUSE)
            gui.typewrite('steam', interval=TYPEWRITE_INT, pause=PAUSE)

            # Run the first option
            gui.press('enter', pause=PAUSE)
            if not __wait_until_image_is_displayed('images/steam_has_loaded.png'):
                continue
            break


def prepare_dota_client():
    while True:
        if _is_dota_launched():
            _focus_dota_window()
            break
        else:
            gui.click(x=STEAM_LIBRARY['x'], y=STEAM_LIBRARY['y'], pause=PAUSE)
            gui.click(x=STEAM_SEARCH['x'], y=STEAM_SEARCH['y'])
            game_search_name = 'dota'
            for i in range(len(game_search_name)):
                gui.press('backspace')
            gui.typewrite(game_search_name, interval=TYPEWRITE_INT)
            if not __wait_until_image_is_displayed('images/dota_is_found.png'):
                continue

            gui.click(x=STEAM_PLAY['x'], y=STEAM_PLAY['y'], pause=30)
            calibrate_dota_client()
            start_game()
            break


def start_game():
    # Leave the game if there is one
    gui.click(x=LEAVE_BUTTON['x'], y=LEAVE_BUTTON['y'], duration=MOUSE_DURATION, pause=PAUSE)
    gui.click(x=MENU_CONFIRM_BUTTON['x'], y=MENU_CONFIRM_BUTTON['y'], pause=4*PAUSE)

    # Start
    gui.click(x=RIGHT_BOT_BUTTON['x'], y=RIGHT_BOT_BUTTON['y'], duration=MOUSE_DURATION, pause=PAUSE)
    gui.click(x=CREATE_LOBBY_BUTTON['x'], y=CREATE_LOBBY_BUTTON['y'], duration=MOUSE_DURATION, pause=PAUSE)
    gui.click(x=JOIN_COACHES_BUTTON['x'], y=JOIN_COACHES_BUTTON['y'], duration=MOUSE_DURATION, pause=PAUSE)
    # Start the game
    gui.click(x=RIGHT_BOT_BUTTON['x'], y=RIGHT_BOT_BUTTON['y'], duration=MOUSE_DURATION, pause=PAUSE)


def restart_game():
    global episodes_since_last_restart
    episodes_since_last_restart += 1
    if episodes_since_last_restart >= RESTART_AFTER_EPISODES:
        episodes_since_last_restart = 0
        close_dota_client()
        close_steam_client()

    # Add a full restart here after a certain number of episodes
    prepare_steam_client()
    prepare_dota_client()

    # Enter the restart command
    gui.press('\\', pause=PAUSE)
    gui.typewrite('restart', interval=TYPEWRITE_INT)
    gui.press('enter', pause=PAUSE)
    gui.press('\\', pause=PAUSE)

    # Wait until the in-game UI is visible
    if not __wait_until_image_is_displayed('images/ingame_arrow.png'):
        restart_game()
        return

    # Start the game right away
    gui.press('\\', pause=PAUSE)
    gui.typewrite('dota_start_game', interval=TYPEWRITE_INT)
    gui.press('enter')
    gui.press('\\', pause=PAUSE)


def close_steam_client():
    if not _is_steam_launched():
        return
    _focus_steam_window()

    gui.click(x=STEAM_MENU['x'], y=STEAM_MENU['y'], pause=PAUSE)
    gui.click(x=STEAM_EXIT['x'], y=STEAM_EXIT['y'], pause=PAUSE)
    time.sleep(30)


def close_dota_client():
    if not _is_dota_launched():
        return
    _focus_dota_window()

    gui.click(x=INGAME_MENU_BUTTON['x'], y=INGAME_MENU_BUTTON['y'], pause=2*PAUSE)
    # Disconnect
    gui.click(x=RIGHT_BOT_BUTTON['x'], y=RIGHT_BOT_BUTTON['y'], pause=2*PAUSE)
    gui.click(x=INGAME_CONFIRM_BUTTON['x'], y=INGAME_CONFIRM_BUTTON['y'], pause=4*PAUSE)
    gui.click(x=EXIT_BUTTON['x'], y=EXIT_BUTTON['y'], pause=2*PAUSE)
    gui.click(x=MENU_CONFIRM_BUTTON['x'], y=MENU_CONFIRM_BUTTON['y'], pause=PAUSE)
    # Wait for complete closure
    time.sleep(30)


def calibrate_dota_client():
    gui.press('\\', pause=PAUSE)
    gui.typewrite('sv_cheats 1', interval=TYPEWRITE_INT)
    gui.press('enter', pause=PAUSE)
    gui.typewrite('host_timescale 5', interval=TYPEWRITE_INT)
    gui.press('enter', pause=PAUSE)
    gui.press('\\', pause=PAUSE)


def __wait_until_image_is_displayed(image_path):
    # Wait for 60 seconds at maximum
    for _ in range(60):
        try:
            point = gui.locateOnScreen(image_path)
            if point:
                return True
        except ImageNotFoundException:
            pass
        time.sleep(1.0)
    return False


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
    time.sleep(0.25)


def _focus_dota_window():
    _run_cmd('wmctrl -a "Dota 2"')
    time.sleep(0.25)


def _run_cmd(cmd):
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return output.decode('utf-8')
