import time
from jet.utils.commands import copy_to_clipboard
from jet.logger import logger
from pynput import mouse, keyboard
from pynput.mouse import Controller, Button
from pynput.keyboard import Key
from typing import Optional

# Initialize mouse controller (Optional type)
mouse_controller: Optional[Controller] = None


def setup_mouse_controller() -> Controller:
    global mouse_controller

    if not mouse_controller:
        mouse_controller = Controller()

    return mouse_controller


def move_to(x: float, y: float, wait_time: float = 0.5) -> None:
    """
    Moves the mouse cursor to (x, y).
    """
    mouse_controller = setup_mouse_controller()

    time.sleep(wait_time)  # Wait a bit to allow window switch

    mouse_controller.position = (x, y)
    logger.log(f'Mouse moved to ({x}, {y})', colors=["GRAY", "DEBUG"])

    time.sleep(wait_time)  # Wait a bit to allow window switch


def move_and_click(x: float, y: float, button: Button = Button.left, wait_time: float = 0.5) -> None:
    """
    Moves the mouse cursor to (x, y), and clicks the specified button.
    """
    mouse_controller = setup_mouse_controller()

    time.sleep(wait_time)  # Wait a bit to allow window switch

    mouse_controller.position = (x, y)
    logger.log(f'Mouse moved to ({x}, {y}) and clicked {button}', colors=[
               "GRAY", "DEBUG"])

    time.sleep(0.2)  # Small delay before clicking
    mouse_controller.click(button)

    time.sleep(wait_time)  # Wait a bit to allow window switch


def start_mouse_listener() -> None:
    mouse_controller = setup_mouse_controller()

    logger.log("Current position:", mouse_controller.position,
               colors=["GRAY", "INFO"])

    def on_move(x: float, y: float) -> None:
        logger.log(f'Mouse moved to ({x}, {y})', colors=["GRAY", "DEBUG"])

    def on_click(x: float, y: float, button: Button, pressed: bool) -> None:
        if pressed:
            logger.log(f'Mouse clicked at ({x}, {y}) with {button}', colors=[
                       "GRAY", "SUCCESS"])

            move_and_click(670.296875, 398.8359375, button)

    def on_scroll(x: float, y: float, dx: float, dy: float) -> None:
        logger.log(f'Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy})', colors=[
                   "GRAY", "DEBUG"])

    def on_press(key: keyboard.Key) -> None:
        try:
            logger.log(f'{key} key pressed.', colors=["GRAY", "INFO"])
            if key == keyboard.Key.enter:
                curr_x, curr_y = mouse_controller.position
                logger.log(f'Enter key pressed. Moving and clicking at ({curr_x}, {curr_y})', colors=[
                           "GRAY", "INFO"])
                move_and_click(curr_x, curr_y, Button.left)
            elif key == keyboard.Key.esc:
                logger.log('Escape key pressed. Exiting...',
                           colors=["GRAY", "INFO"])
                return False  # Stops the listener if ESC is pressed
            elif key == keyboard.Key.space:
                logger.log('Space key pressed.', colors=["GRAY", "INFO"])
            elif key == keyboard.Key.f1:
                logger.log('F1 key pressed.', colors=["GRAY", "INFO"])
            elif hasattr(key, 'char') and key.char == 'a':
                logger.log('A key pressed.', colors=["GRAY", "INFO"])
            elif hasattr(key, 'char') and key.char == 'b':
                logger.log('B key pressed.', colors=["GRAY", "INFO"])
            elif key == keyboard.Key.cmd:  # Detect 'Cmd'
                # if hasattr(key, 'char') and key.char == 'c':  # Detect 'C'
                curr_x, curr_y = mouse_controller.position
                coords = f"({curr_x}, {curr_y})"
                # copy_to_clipboard(coords)  # Copy coordinates to clipboard
                logger.orange(f'Coordinates {coords}')

        except AttributeError:
            pass  # Ignore special keys other than the ones we check

    # Set up the listeners for both mouse and keyboard
    with mouse.Listener(
        # on_move=on_move,
        # on_click=on_click,
        on_scroll=on_scroll) as mouse_listener, \
            keyboard.Listener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join()  # Listen for keyboard events
