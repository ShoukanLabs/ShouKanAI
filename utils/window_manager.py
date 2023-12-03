import cv2
import numpy as np
import pyautogui
import pygetwindow
import pygetwindow as gw


def get_window_frame(window_name="VRChat", window_size=(640, 480)):
    # Set up window
    window = gw.getWindowsWithTitle(window_name)[0]
    try:
        window.activate()
    except pygetwindow.PyGetWindowException:
        pass
    window.size = window_size

    # Capture frame
    img = pyautogui.screenshot(region=(window.left + 10, window.top + 40, window.width - 20, window.height - 50))
    frame = np.array(img)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    while True:
        frame = get_window_frame()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
