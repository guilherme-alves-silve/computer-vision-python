import cv2
import mediapipe as mp
import time

import numpy as np


class FaceDetector:

    def __init__(self,
                 min_detection_confidence=0.5,
                 model_selection=0):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            min_detection_confidence,
            model_selection
        )

    def find_faces(self, img: np.ndarray, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)

        bboxes = []

        if results.detections:

            for id, detection in enumerate(results.detections):
                # mp_draw.draw_detection(img, detection)
                # or
                mp_bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(mp_bbox.xmin * w), int(mp_bbox.ymin * h), int(mp_bbox.width * w), int(mp_bbox.height * h)

                bboxes.append([id, bbox, detection.score])
                if draw:
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 255, 0), 2)
                    img = self._fancy_draw(img, bbox)

        return img, bboxes

    def _fancy_draw(self, img: np.ndarray, bbox, length=30, thickness=3):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # Top Left x, y
        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1-length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y+length), (255, 0, 255), thickness)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x+length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1-length), (255, 0, 255), thickness)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1-length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1-length), (255, 0, 255), thickness)
        return img


def main():
    prev_time = 0
    curr_time = 0

    face_detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img, bboxes = face_detector.find_faces(img)

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
