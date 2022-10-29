from typing import NamedTuple

import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector:

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.results = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode,
            max_num_hands,
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self,
                   img: np.ndarray,
                   draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self,
                      img: np.ndarray,
                      hand_no=0,
                      hand_part_id=0,
                      draw=True):

        landmarks = []
        if not self.results.multi_hand_landmarks:
            return landmarks

        hand_landmark = self.results.multi_hand_landmarks[hand_no]

        for id, landmark in enumerate(hand_landmark.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([id, cx, cy])
            if draw and id == hand_part_id:
                cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return landmarks


def main():
    prev_time = 0
    curr_time = 0

    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = hand_detector.find_hands(img)
        hand_detector.find_position(img)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
