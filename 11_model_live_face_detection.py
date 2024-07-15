import cv2
import os

face_config = os.path.join(
    os.path.dirname(__file__),
    'models',
    'cascade',
    'haarcascade_frontalface_default.xml'
)
eye_config = os.path.join(
    os.path.dirname(__file__),
    'models',
    'cascade',
    'haarcascade_eye.xml'
)
face_detector = cv2.CascadeClassifier(face_config)
eye_detector = cv2.CascadeClassifier(eye_config)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    detections = eye_detector.detectMultiScale(image_gray)
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()