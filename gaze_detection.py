import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
from skimage.transform import resize
from skimage.io import imread


currentData = []
dirData = []
count = 0
timer = 60
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# variable to load machine learning model
CNNModel = load_model('cnnModel.keras')

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    # flip image to get mirror image
    img = cv2.cvtColor(cv2.flip(img,1), cv2.COLOR_BGR2RGB)
    # improve performance during face mesh processing
    img.flags.writeable = False
    results = face_mesh.process(img)
    img.flags.writeable = True

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = img.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x* img_w, lm.y* img_h)
                        nose_3d = (lm.x* img_w, lm.y* img_h, lm.z * 3000)

                    x,y = int(lm.x* img_w), int(lm.y* img_h)

                    # get coordinates
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])

            # convert to numpy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # camera matrix
            focal_length = 1* img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
            # distortion matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] *360
            y = angles[1] * 360
            z = angles[2] * 360

            # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(img, p1, p2, (255,0,0), 3)

        mp_drawing.draw_landmarks(
                image=img,
                landmark_list = face_landmarks,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose Estimation', img)

    timer -= 1
    if timer <= 0:
        cv2.imwrite('currentData/photo.jpg', img)
        predictImg = imread('currentData/photo.jpg')
        predictImg = tf.image.resize(predictImg, (256, 256))
        direction = np.argmax(CNNModel.predict(np.expand_dims(predictImg/255, 0)))

        if direction == 0:
            print('center')
        elif direction == 1:
            print('down')
        elif direction == 2:
            print('left')
        elif direction == 3:
            print('right')
        elif direction == 4:
            print('up')
        timer = 60

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
