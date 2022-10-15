import cv2


def init_trackbars(window_name="TrackBars"):
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 640, 240)
    cv2.createTrackbar("Hue Min", window_name, 0, 179, lambda value: None)
    cv2.createTrackbar("Hue Max", window_name, 19, 179, lambda value: None)
    cv2.createTrackbar("Saturation Min", window_name, 110, 255, lambda value: None)
    cv2.createTrackbar("Saturation Max", window_name, 240, 255, lambda value: None)
    cv2.createTrackbar("Value Min", window_name, 153, 255, lambda value: None)
    cv2.createTrackbar("Value Max", window_name, 255, 255, lambda value: None)


def get_trackbars_values(window_name="TrackBars"):
    hue_min = cv2.getTrackbarPos("Hue Min", window_name)
    hue_max = cv2.getTrackbarPos("Hue Max", window_name)
    saturation_min = cv2.getTrackbarPos("Saturation Min", window_name)
    saturation_max = cv2.getTrackbarPos("Saturation Max", window_name)
    value_min = cv2.getTrackbarPos("Value Min", window_name)
    value_max = cv2.getTrackbarPos("Value Max", window_name)
    return hue_min, hue_max, saturation_min, saturation_max, value_min, value_max
