import cv2
import numpy as np
import pyautogui
import time

from src.data.data_container import DataContainer

# Define the application window title (modify this to match your Vortex window title)
# vortex_window_titles = ["Vortex", 'Chrome']
# vortex_window_title = "Vortex"
templates = ['img.png', 'img_1.png']
while True:
    try:
        # Check if the Vortex application is in focus
        if pyautogui.getActiveWindow().title:
            # Load the screenshot of the screen
            screenshot = pyautogui.screenshot()
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            for path in templates:
                # Load the image of the button (the template)
                template = cv2.imread('imgs/' + path, cv2.IMREAD_COLOR)  # Replace with your button image file

                # Perform template matching
                result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Get the coordinates of the best match
                button_x, button_y = max_loc

                # Click the button
                if max_val > 0.95:
                    pyautogui.click(button_x + 10, button_y + 10)
                    pyautogui.click(button_x + 10, button_y + 10)
                    print(f"Clicked at ({button_x}, {button_y}). Val {max_val}, Temp {path}")
                    time.sleep(1)
    except:
        continue
