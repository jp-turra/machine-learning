import os
import cv2
import cvzone
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from ultralytics import YOLO, solutions, SAM
from ultralytics.utils.plotting import Annotator, colors

def initialize_video_capture(source: os.PathLike | int = 0, width: int = 960, height: int = 840, fps: int = 30) -> cv2.VideoCapture:
    """Initialize and configure the video capture device."""
    
    webcam = cv2.VideoCapture(source)

    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if type(source) == int:
        webcam.set(cv2.CAP_PROP_FPS, fps)
        webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    return webcam

def get_yolo_for_tensorflow(model_name: str, **kwargs) -> YOLO:
    if not os.path.exists(f"{model_name}_saved_model"):
        yolo = YOLO(model_name, **kwargs)
        yolo.export(format="saved_model", keras=True)

    yolo = YOLO(f"{model_name}_saved_model", **kwargs)
    return yolo

test_image = 'datasets/playground/curso/Images/car.jpg'

# yolo10n = get_yolo_for_tensorflow("yolov10n", task='detect')
# yolo10_labels = yolo10n.names

# yolo8seg = get_yolo_for_tensorflow("yolov8s-seg", task='segment')
# yolo8seg_labels = yolo8seg.names

yolo8seg = SAM('mobile_sam.pt')
yolo8seg_labels = yolo8seg.names

# yolo8 = get_yolo_for_tensorflow("yolov8n", task='detect')
# yolo8_labels = yolo8.names

line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=yolo8seg_labels,
    view_img=True,
)

webcam = initialize_video_capture(source=0)
# webcam = initialize_video_capture('datasets/playground/curso/Videos/race.mp4')

if not webcam.isOpened():
    print("Could not open the camera!")
else:
    while webcam.isOpened():
        try:
            ret, frame = webcam.read()

            annotator = Annotator(frame, line_width=2)
            results = yolo8seg.track(frame, persist=True, conf=0.5, device='cuda')

            # frame = speed_obj.estimate_speed(frame, results)

            if not ret:
                print("Something went wrong reading the video!")
                break

            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Frame is empty!")
                break

            if results[0].boxes.id is not None and results[0].masks is not None:
                masks = results[0].masks.xy
                track_ids = results[0].boxes.id.int().cpu().tolist()
                cls = results[0].boxes.cls.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()

                for mask, track_id, label, conf in zip(masks, track_ids, cls, confs):
                    color = colors(int(track_id), True)
                    txt_color = annotator.get_txt_color(color)
                    annotator.seg_bbox(mask=mask, mask_color=color, label=f"[{track_id}] {yolo8seg_labels[label]} - {conf:.2f}", txt_color=txt_color)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    webcam.release()
    cv2.destroyAllWindows()