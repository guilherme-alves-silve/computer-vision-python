import cv2
import numpy as np
from enum import Enum


class Shapes(Enum):
    TRIANGLE = 3
    SQUARE = 4
    CIRCLE = 8


def pre_processing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1)
    img_canny = cv2.Canny(img_blur, 200, 200)

    kernel = np.ones((5, 5))
    img_dialation = cv2.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv2.erode(img_dialation, kernel, iterations=1)

    return img_threshold


def get_contours(reference_img, img_result=None, min_threshold=0, points_shape=Shapes.SQUARE, test=False):
    # cv2.RETR_EXTERNAL: return the extreme outer contours
    # cv2.CHAIN_APPROX_NONE: get all the contours we have found
    contours, hierarchy = cv2.findContours(reference_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_all_contours = -1

    max_area = 0
    biggest_approx = np.array([])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_threshold:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
            if area > max_area and (len(approx) == points_shape.value):
                max_area = area
                biggest_approx = approx
                if test:
                    assert img_result is not None, "The img_result can't be none when testing"
                    cv2.drawContours(img_result, contour, draw_all_contours,
                                     color=(255, 0, 0), thickness=3)
                    cv2.drawContours(img_result, biggest_approx, draw_all_contours,
                                     color=(255, 0, 0), thickness=20)
    return biggest_approx


def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), np.int32)

    add = points.sum(axis=1)
    points_new[0] = points[np.argmin(add)]
    points_new[-1] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]

    return points_new


def get_warp(img, approx, width, height, crop=20):
    pts1 = np.float32(reorder(approx))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (width, height))

    if crop > 0:
        img_cropped = img_warped[crop:img_warped.shape[0]-crop, crop:img_warped.shape[1]-crop]
        img_cropped = cv2.resize(img_cropped, (width, height))
        return img_cropped

    return img_warped
