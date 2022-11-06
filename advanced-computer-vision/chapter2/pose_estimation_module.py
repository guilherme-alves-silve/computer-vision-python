import cv2
import mediapipe as mp
import time
import numpy as np


class PoseDetector:

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode,
            model_complexity,
            smooth_landmarks,
            enable_segmentation,
            smooth_segmentation,
            min_detection_confidence,
            min_tracking_confidence
        )

    def find_pose(self, img: np.ndarray, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img: np.ndarray, draw=True):

        landmarks = []

        if not self.results.pose_landmarks:
            return landmarks

        for id, landmark in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmarks.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return landmarks

    def draw_landmark(self, img, landmarks, body_part_num: int):
        if not landmarks:
            return

        elbow_landmark = landmarks[body_part_num]
        cv2.circle(img, (elbow_landmark[1], elbow_landmark[2]), 15, (0, 0, 255), cv2.FILLED)


def main():
    prev_time = 0
    curr_time = 0

    pose_detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = pose_detector.find_pose(img)
        landmarks = pose_detector.find_position(img)
        elbow = 14
        pose_detector.draw_landmark(img, landmarks, elbow)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("Video", img)


if __name__ == "__main__":
    main()
