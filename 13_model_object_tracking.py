# Rastreamento x Detecção de Objetos

## * Ratreamento consiste em detectar o objeto e rastrear o movimento do objeto em um vídeo
## * Detecção é apenas para informar se um objeto existe um frame

## * https://medium.com/@khwabkalra1/object-tracking-2fe4127e58bf

# KCF (Kernel Correlation Filters)

## * Utiliza filtro de partícula para ajustar o bounding boxes a cada frame.
## * Utilizado para detecção rápida, porém apresenta qualidade/precisão de detecção menor quando comparado com o CSRT

# CSRT (Discriminative Corralation Filter with Channel and Spatial Reliability)

## * Mais lento porém mais preciso que o KCF

### * https://medium.com/@dufresne.danny/usb-webcam-wls2-setup-gvcuview-cheese-opencv-c-b4b5bc43df29

import cv2
import time
import os

USE_WEBCAM = 1
USE_CSRT = 1

if USE_WEBCAM:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
else:
    race_video_path = os.path.join(
        os.getcwd(), 'datasets', 'playground', 'curso', 'Videos', 'race.mp4'
    )
    street_video_path = os.path.join(
        os.getcwd(), 'datasets', 'playground', 'curso', 'Videos', 'street.mp4'
    )
    video_capture = cv2.VideoCapture(street_video_path)

if not video_capture.isOpened():
    print("Could not open the camera!")
else:
    if USE_CSRT:
        tracking = cv2.TrackerCSRT_create()
    else:
        tracking = cv2.TrackerKCF_create()

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        print("Something went wrong reading the video!")
    else:
        # Selecting the Region Of Interest (ROI)
        bbox = cv2.selectROI("SELECT_ROI", frame)
        cv2.destroyWindow("SELECT_ROI")
        # print(bbox)

        # Initialize the tracking with the selected object
        tracking.init(frame, bbox)

        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("Something went wrong reading the video!")
                break

            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Frame is empty!")
                break
            
            # Update the ROI
            ret, bbox = tracking.update(frame)
            if ret:
                (x, y, w, h) = [int(item) for item in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)
            else:
                print("Something went wrong updating the ROI!")

            # Display the resulting frame
            cv2.imshow('Rastreamento', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # input("Press any key to continue...")
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyWindow("Rastreamento")