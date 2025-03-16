import cv2
from imread_from_url import imread_from_url
import os
import psutil
import numpy as np
from tkinter import filedialog, Tk, messagebox
from PIL import Image, ImageTk

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

def run_image_multipose_estimation(app):
    def get_image_path():
        root = Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png ")])
        root.destroy()
        return file_path

    file_path = get_image_path()
    if file_path:
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            messagebox.showerror("Error", "Failed to read the image file. Please check the file path and integrity.")
            return
    else:
        return

    # Initialize Pose Estimation model
    model_path = "models/hrnet_coco_w48_384x288.onnx"
    model_type = ModelType.COCO
    hrnet = HRNET(model_path, model_type, conf_thres=0.5)

    # Initialize Person Detection model
    person_detector_path = "models/yolov5s6.onnx"
    person_detector = PersonDetector(person_detector_path)

    # Detect People in the image
    detections = person_detector(img)
    ret, person_detections = filter_person_detections(detections)

    if ret:
        boxes, scores, class_ids = person_detections

        # Estimate the pose in the image
        total_heatmap, poses, angles = hrnet(img, person_detections)

        # Draw Model Output
        img = hrnet.draw_pose(img)

        # Update person selector options
        app.update_person_selector(len(hrnet.poses))

        # Draw detections
        img = hrnet.draw_bounding_boxes(img, (boxes, scores, class_ids))

        # Display the result in the GUI result window
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((app.image_label.winfo_width(), app.image_label.winfo_height()), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        app.update_image_label(img_tk)

        # Save the image
        try:
            cv2.imwrite("doc/img/output.jpg", cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error saving image: {e}")

        # Store angles for later use
        app.angles = angles
        app.update_angles()
    else:
        messagebox.showinfo("Info", "No person detected in the image.")

if __name__ == "__main__":
    run_image_multipose_estimation(None)
