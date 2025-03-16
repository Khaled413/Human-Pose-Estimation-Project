import cv2
from imread_from_url import imread_from_url
from tkinter import filedialog, Tk, messagebox, StringVar, OptionMenu
import numpy as np
from PIL import Image, ImageTk  # Ensure ImageTk is imported

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

def run_image_singlepose_estimation(app):
    def get_image_path():
        root = Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        root.destroy()
        return file_path

    def update_person_selector(num_persons):
        options = [f"Person {i+1}" for i in range(num_persons)]
        selected_person.set(options[0] if options else "")
        app.person_selector['menu'].delete(0, 'end')
        for option in options:
            app.person_selector['menu'].add_command(label=option, command=lambda value=option: selected_person.set(value))

    def display_image(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((app.image_label.winfo_width(), app.image_label.winfo_height()), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        app.image_label.configure(image=img_tk)
        app.image_label.image = img_tk

    def display_angles(angles):
        if angles:
            angle_text = "Joint Angles:\n"
            for joint, angle in angles.items():
                if angle is not None:
                    angle_text += f"{joint}: {angle:.2f}°\n"
            app.angle_text.configure(text=angle_text)
        else:
            app.angle_text.configure(text="No angles detected.")

    file_path = get_image_path()
    if file_path:
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            messagebox.showerror("Error", "Failed to read the image file. Please check the file path and integrity.")
            return
    else:
        return

    # Clear any existing person selector
    if hasattr(app, 'person_selector'):
        app.person_selector.pack_forget()

    # Initialize models
    model_path = "models/hrnet_coco_w48_384x288.onnx"
    model_type = ModelType.COCO
    hrnet = HRNET(model_path, model_type, conf_thres=0.5)

    person_detector_path = "models/yolov5s6.onnx"
    person_detector = PersonDetector(person_detector_path)

    # Detect people in the image
    detections = person_detector(img)
    ret, person_detections = filter_person_detections(detections)

    if ret and len(person_detections[0]) > 0:
        boxes, scores, class_ids = person_detections

        if len(boxes) == 1:
            # Estimate the pose for the single detected person
            total_heatmap, poses, angles = hrnet(img, person_detections)
            output_img = hrnet.draw_pose(img)
            output_img = hrnet.draw_bounding_boxes(output_img, person_detections)
            display_image(output_img)
            display_angles(angles[0])
        else:
            # Multiple people detected, create a selection list
            selected_person = StringVar(app)
            app.person_selector = OptionMenu(app.button_frame, selected_person, "")
            app.person_selector.configure(font=("Arial", 20, "bold"), bg="#FFFFFF", fg="#000000", borderwidth=2, relief="groove")
            app.person_selector.pack(pady=10, padx=10, fill="x")

            update_person_selector(len(boxes))

            def on_person_selected(*args):
                selected_index = int(selected_person.get().split()[-1]) - 1
                selected_box = [boxes[selected_index]]
                selected_scores = [scores[selected_index]]
                selected_class_ids = [class_ids[selected_index]]
                selected_detections = [selected_box, selected_scores, selected_class_ids]

                total_heatmap, poses, angles = hrnet(img, selected_detections)
                output_img = hrnet.draw_pose(img)
                output_img = hrnet.draw_bounding_boxes(output_img, selected_detections)
                display_image(output_img)
                display_angles(angles[0])

            selected_person.trace("w", on_person_selected)

            # Set default selection to Person 1 and trigger the callback
            selected_person.set("Person 1")
            on_person_selected()
    else:
        messagebox.showinfo("Info", "No person detected in the image.")

if __name__ == "__main__":
    run_image_singlepose_estimation(None)
