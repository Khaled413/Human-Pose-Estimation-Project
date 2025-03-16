import cv2
import os
import psutil
import threading
from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections
from PIL import Image, ImageTk
import customtkinter as ctk

# Initialize video from a local file
video_path = r"F:\Project\ONNX-HRNET-Human-Pose-Estimation\hrnet_video.gif"  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 720))

# Initialize Pose Estimation model
model_path = "models/hrnet_coco_w48_384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.3)

# Initialize Person Detection model
person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path, conf_thres=0.3)

frame_num = 0

def run_video_multipose_estimation(app):
    global cap, out, frame_num
    while cap.isOpened() and app.video_running:
        # Press key q to stop or close the window to stop
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            # Read frame from the video
            ret, frame = cap.read()

            # Skip the first {start_time} seconds
            if frame_num < start_time * 30:
                frame_num += 1
                continue

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video from the beginning
                continue
        except Exception as e:
            print(e)
            continue

        # Detect People in the image
        detections = person_detector(frame)
        ret, person_detections = filter_person_detections(detections)
        person_detector.boxes, person_detector.scores, person_detector.class_ids = person_detections

        if ret:
            # Estimate the pose in the image
            total_heatmap, peaks, angles = hrnet(frame, person_detections)

            # Draw Model Output
            frame = hrnet.draw_pose(frame)
            frame = person_detector.draw_detections(frame, mask_alpha=0.15)

        # Display the frame in the GUI result window
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img.resize((app.result_frame.winfo_width(), app.result_frame.winfo_height()), Image.Resampling.LANCZOS))
        app.update_image_label(img_tk)

        out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
