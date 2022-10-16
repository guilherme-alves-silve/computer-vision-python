import cv2
import numpy as np
from stack_images import stack_images


def get_colors_pen():
    """
    Output:
        h = hue
        s = saturation
        v = value
        [h_min, s_min, v_min, h_max, s_max, v_max]
    """
    return [[87, 84, 41, 154, 255, 192],  # blue pen
            [140, 112, 74, 179, 255, 255],  # red pen
            [45, 54, 72, 146, 255, 255]]  # purple pen


def get_color_values():
    return [[255, 0, 0],  # blue bgr
            [0, 0, 255],  # red bgr
            [255, 0, 255]]  # purple bgr


def find_color(img, img_result, colors=None, colors_values=None, test=False):

    if colors is None:
        colors = get_colors_pen()

    if colors_values is None:
        colors_values = get_color_values()

    img_colors = []
    new_points = []
    for i in range(len(colors)):
        color = colors[i]
        color_value = colors_values[i]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(img_hsv, lower, upper)
        x, y = draw_contours(mask, img_result, test)
        cv2.circle(img_result, (x, y), 10, color_value, cv2.FILLED)
        if x != 0 and y != 0:
            new_points.append([x, y, i])

        if test:
            img_colors.append(img_result)

    if test:
        imgs = stack_images(0.6, (img_colors,))
        cv2.imshow("Images", imgs)

    return new_points


def draw_on_canvas(points, img_result, colors_values=None):

    if not colors_values:
        colors_values = get_color_values()

    for point in points:
        cv2.circle(img_result, (point[0], point[1]), 10, colors_values[point[2]], cv2.FILLED)


def draw_contours(reference_img, img_draw_contours, min_threshold=0, test=False):
    # cv2.RETR_EXTERNAL: return the extreme outer contours
    # cv2.CHAIN_APPROX_NONE: get all the contours we have found
    contours, hierarchy = cv2.findContours(reference_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_all_contours = -1

    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_threshold:
            if test:
                cv2.drawContours(img_draw_contours, contour, draw_all_contours,
                                 color=(255, 0, 0), thickness=3)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y
