import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)

curr_time = 0
prev_time = 0.1

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmark, mp_face_mesh.FACEMESH_CONTOURS,
                                   draw_spec, draw_spec)
            for id, landmark in enumerate(face_landmark.landmark):
                ih, iw, ic = landmark.shape
                x, y = int(landmark.x*iw), int(landmark.y*ih)


    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Video", img)
