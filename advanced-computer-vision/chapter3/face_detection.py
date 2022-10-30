import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()

cap = cv2.VideoCapture(0)

curr_time = 0
prev_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(img_rgb)

    if results.detections:

        for id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            # or
            mp_bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(mp_bbox.xmin * w), int(mp_bbox.ymin * h), \
                   int(mp_bbox.width * w), int(mp_bbox.height * h)

            cv2.rectangle(img, bbox, (255, 0, 255), 3)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Video", img)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
