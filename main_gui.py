import customtkinter as ctk
from tkinter import messagebox, StringVar, OptionMenu, Tk, filedialog
from imread_from_url import imread_from_url
from PIL import Image, ImageTk 
import subprocess
import cv2
import threading
import os
import signal
import numpy as np 
from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections                                    
from video_multipose_estimation import run_video_multipose_estimation
from image_multipose_estimation import run_image_multipose_estimation
from image_singlepose_estimation import run_image_singlepose_estimation

class PoseEstimationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Human-Pose-Estimation")
        self.geometry("1200x800+{}+{}".format(self.winfo_screenwidth()//2 - 600, 0))
        self.configure(bg="#F3F4F6")
        self.iconbitmap("icon.ico")  # Set the icon
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.selected_person = StringVar(self)
        self.selected_person.trace("w", self.update_angles)
        self.angles = None  # Initialize self.angles
        self.create_widgets()
        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.webcam_running = False
        self.video_running = False
        self.active_module = None  # Track the currently active module
        # Initialize Pose Estimation model
        model_path = "models/hrnet_coco_w48_384x288.onnx"
        model_type = ModelType.COCO
        self.hrnet = HRNET(model_path, model_type, conf_thres=0.5)

        # Initialize Person Detection model
        person_detector_path = "models/yolov5s6.onnx"
        self.person_detector = PersonDetector(person_detector_path)

    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="#F3F4F6")
        self.main_frame.pack(fill=ctk.BOTH, expand=True)

        self.title_label = ctk.CTkLabel(self.main_frame, text="Human Pose Estimation", font=("Arial", 50, "bold"), fg_color="#2C3E50", text_color="#FFFFFF")
        self.title_label.pack(fill=ctk.X, pady=20)

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="#34495E", width=100)
        self.button_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=20, pady=0)

        self.result_frame = ctk.CTkFrame(self.main_frame, fg_color="#ECF0F1")
        self.result_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=1, pady=0)

        buttons = [
            ("Single-person Image Pose Estimation", self.run_image_singlepose),
            ("Multi-person Image Pose Estimation", self.run_image_multipose),
            ("Multi-person Video Pose Estimation", self.run_video_multipose),
            ("Multi-person Webcam Pose Estimation", self.run_webcam_multipose)
        ]

        self.button_widgets = []  # Store button widgets for updating GUI
        for text, command in buttons:
            button = ctk.CTkButton(self.button_frame, text=text, command=command, fg_color="#FFFFFF", hover_color="#DDDDDD", text_color="#000000", font=("Arial", 20, "bold"))
            button.pack(pady=15, padx=10, fill="x")
            self.button_widgets.append(button)

        self.person_selector = OptionMenu(self.button_frame, self.selected_person, "")
        self.person_selector.configure(font=("Arial", 20, "bold"), bg="#FFFFFF", fg="#000000", borderwidth=2, relief="groove")
        self.person_selector.pack_forget()  # Hide the selector initially

        self.angle_text = ctk.CTkLabel(self.button_frame, text="", font=("Arial", 20, "bold"), fg_color="#34495E", text_color="#FFFFFF")
        self.angle_text.pack_forget()  # Hide the angle text initially

        self.result_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 25, "bold"), fg_color="#ECF0F1", text_color="#2C3E50")
        self.result_label.pack(pady=10)

        self.image_label = ctk.CTkLabel(self.result_frame, fg_color="#ECF0F1", text="")
        self.image_label.pack(pady=15, fill=ctk.BOTH, expand=True)

        self.select_image_button = ctk.CTkButton(self.button_frame, text="Select Image", command=self.select_image, fg_color="#FFFFFF", hover_color="#DDDDDD", text_color="#000000", font=("Arial", 20, "bold"))
        self.select_image_button.pack_forget()  # Hide the button initially

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                messagebox.showerror("Error", "Failed to read the image file. Please check the file path and integrity.")
                return

            # Detect People in the image
            detections = self.person_detector(img)
            ret, person_detections = filter_person_detections(detections)

            if ret:
                # Estimate the pose in the image
                total_heatmap, poses, angles = self.hrnet(img, person_detections)

                # Draw Model Output
                img = self.hrnet.draw_pose(img)
                img = self.hrnet.draw_bounding_boxes(img, person_detections)

                # Display the result in the GUI result window
                self.display_image(img)

                # Store angles for later use
                self.angles = angles
                self.update_angles()
            else:
                messagebox.showinfo("Info", "No person detected in the image.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Get the dimensions of the image and the result frame
        img_width, img_height = img_pil.size
        frame_width = self.result_frame.winfo_width()
        frame_height = self.result_frame.winfo_height()

        # Calculate the scaling factor to fit the image within the frame while maintaining aspect ratio
        scale = min(frame_width / img_width, frame_height / img_height, 1.0)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize the image
        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the image label size and display the image
        self.image_label.configure(width=new_width, height=new_height, image=img_tk)
        self.image_label.image = img_tk
        self.image_label.pack_propagate(False)
        self.image_label.update_idletasks()
        self.image_label.update()

    def run_image_singlepose(self):
        self.stop_all()
        self.clear_image()
        self.result_label.configure(text="Result of image_singlepose_estimation.py")  # Ensure this text appears
        self.select_image_button.pack(pady=15, padx=10, fill="x")  # Show the button
        self.angle_text.pack(pady=10, padx=10, fill="x")  # Show the angle text
        self.active_module = "image_singlepose"
        self.update_active_button()
        threading.Thread(target=run_image_singlepose_estimation, args=(self,)).start()

    def run_image_multipose(self):
        self.stop_all()
        self.clear_image()
        self.result_label.configure(text="Result of image_multipose_estimation.py")  # Ensure this text appears
        self.select_image_button.pack(pady=15, padx=10, fill="x")  # Show the button
        self.person_selector.pack(pady=10, padx=10, fill="x")  # Show the selector
        self.angle_text.pack(pady=10, padx=10, fill="x")  # Show the angle text
        self.active_module = "image_multipose"
        self.update_active_button()
        self.selected_person.set("")  # Reset the person selector
        threading.Thread(target=self.process_multipose_image).start()

    def process_singlepose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
        if file_path:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                messagebox.showerror("Error", "Failed to read the image file. Please check the file path and integrity.")
                return

            # Detect People in the image
            detections = self.person_detector(img)
            ret, person_detections = filter_person_detections(detections)
            if ret and len(person_detections[0]) > 0:
                # Estimate the pose for the first detected person
                total_heatmap, poses, angles = self.hrnet(img, person_detections)

                # Draw model output
                output_img = self.hrnet.draw_pose(img)
                output_img = self.hrnet.draw_bounding_boxes(output_img, person_detections)

                # Display the result
                self.display_image(output_img)

                # Display angles
                if angles:
                    angle_text = "Joint Angles:\n"
                    for joint, angle in angles[0].items():
                        if angle is not None:
                            angle_text += f"{joint}: {angle:.2f}°\n"
                    self.angle_text.configure(text=angle_text)
                else:
                    self.angle_text.configure(text="No angles detected.")
            else:
                messagebox.showinfo("Info", "No person detected in the image.")
        else:
            messagebox.showinfo("Info", "No file selected.")
    def process_multipose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                messagebox.showerror("Error", "Failed to read the image file. Please check the file path and integrity.")
                return
            # Detect People in the image
            detections = self.person_detector(img)
            ret, person_detections = filter_person_detections(detections)
            if ret:
                boxes, scores, class_ids = person_detections
                # Estimate the pose in the image
                total_heatmap, poses, angles = self.hrnet(img, person_detections)
                # Draw Model Output
                img = self.hrnet.draw_pose(img)
                # Update person selector options
                self.update_person_selector(len(self.hrnet.poses))

                # Draw detections
                img = self.hrnet.draw_bounding_boxes(img, (boxes, scores, class_ids))

                # Display the result in the GUI result window
                self.display_image(img)
                # Store angles for later use
                self.angles = angles
                self.update_angles()
            else:
                messagebox.showinfo("Info", "No person detected in the image.")
        else:
            messagebox.showinfo("Info", "No file selected.")
    def run_video_multipose(self):
        self.stop_all()
        self.clear_image()
        self.video_running = True
        self.result_label.configure(text="Result of video_multipose_estimation.py")
        self.person_selector.pack_forget()  # Hide the selector
        self.angle_text.pack_forget()  # Hide the angle text
        self.select_image_button.pack_forget()  # Hide the button
        self.active_module = "video_multipose"
        self.update_active_button()
        self.selected_person.set("")  # Reset the person selector
        threading.Thread(target=run_video_multipose_estimation, args=(self,)).start()
    def run_webcam_multipose(self):
        self.stop_all()
        self.clear_image()
        self.webcam_running = True
        self.result_label.configure(text="Result of webcam_multipose_estimation.py")
        self.person_selector.pack(pady=10, padx=10, fill="x")  # Show the selector
        self.select_image_button.pack_forget()  # Hide the button
        self.angle_text.pack(pady=10, fill="x")  # Show the angle text
        self.active_module = "webcam_multipose"
        self.update_active_button()
        self.selected_person.set("")  # Reset the person selector
        threading.Thread(target=self.run_webcam_script).start()
    def stop_all(self):
        self.webcam_running = False
        self.video_running = False
        self.clear_angle_text()
        cv2.destroyAllWindows()
        self.active_module = None
        self.update_active_button()
        # Clear any existing person selector
        if hasattr(self, 'person_selector'):
            self.person_selector.pack_forget()
    def clear_image(self):
        self.image_label.configure(image='')
        self.image_label.image = None
    def run_script(self, script_name, is_video=False):
        self.stop_all()
        self.clear_image()
        if is_video:
            threading.Thread(target=self.run_video_script, args=(script_name,)).start()
        else:
            try:
                subprocess.run(["python", script_name], check=True)
                self.display_result(script_name)
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Error", f"Failed to run {script_name}: {e}")
    def run_image_multipose_script(self):
        subprocess.run(["python", "image_multipose_estimation.py"], check=True)
        self.display_result("image_multipose_estimation.py")
        self.display_angles(self.hrnet, self.person_detector)
    def run_video_script(self, script_name):
        if script_name == "video_multipose_estimation.py":
            self.run_video_multipose()
        else:
            subprocess.run(["python", script_name], check=True)
            cap = cv2.VideoCapture("output.avi")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(img.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS))
                self.update_image_label(img_tk)
            cap.release()
    def run_webcam_script(self):
        # Initialize Person Detection model
        person_detector_path = "models/yolov5s6.onnx"
        person_detector = PersonDetector(person_detector_path)
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and self.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break
            # Detect People in the image
            detections = person_detector(frame)
            ret, person_detections = filter_person_detections(detections)
            # Estimate the pose in the image
            total_heatmap, peaks, angles = self.hrnet(frame, person_detections)
            if ret:
                # Draw Model Output
                output_img = self.hrnet.draw_pose(frame)
                output_img = self.hrnet.draw_bounding_boxes(output_img, person_detections)
                # Update person selector options
                self.update_person_selector(len(person_detections[0]))
                # Update angle text in the button frame
                self.angles = angles
                self.update_angles()
                img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS))
                self.update_image_label(img_tk)
            if not self.webcam_running:
                break
        cap.release()
    def update_person_selector(self, num_persons):
        options = [f"Person {i+1}" for i in range(num_persons)]
        self.selected_person.set(options[0] if options else "")
        self.person_selector['menu'].delete(0, 'end')
        for option in options:
            self.person_selector['menu'].add_command(label=option, command=lambda value=option: self.selected_person.set(value))
        self.selected_person.set(options[0] if options else "")  # Ensure the dropdown shows the correct selection
    def update_angles(self, *args):
        selected_index = int(self.selected_person.get().split()[-1]) - 1 if self.selected_person.get() else None
        if selected_index is not None and self.angles:
            angle_text = f"Angles for {self.selected_person.get()}:\n"
            for joint, angle in self.angles[selected_index].items():
                if angle is not None:
                    angle_text += f"{joint}: {angle:.2f}\n"
            self.angle_text.configure(text=angle_text)
    def display_angles(self, hrnet, person_detector):
        # Read image
        img_url = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Flickr_-_The_U.S._Army_-_%27cavalry_charge%27.jpg"
        img = imread_from_url(img_url)
        # Detect People in the image
        detections = person_detector(img)
        ret, person_detections = filter_person_detections(detections)
        if ret:
            boxes, scores, class_ids = person_detections
            # Estimate the pose in the image
            total_heatmap, peaks, angles = hrnet(img, person_detections)
            # Calculate and display joint angles
            angle_text = ""
            for i, pose in enumerate(hrnet.poses):
                pose_angles = hrnet.calculate_angles(pose)
                angle_text += f"Person {i+1}:\n"
                for joint, angle in pose_angles.items():
                    if angle is not None:
                        angle_text += f"{joint}: {angle:.2f}\n"
            self.angle_text.configure(text=angle_text)
    def update_image_label(self, img_tk):
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.update_idletasks()
        self.update()
    def display_result(self, script_name):
        self.result_label.configure(text=f"Result of {script_name}")  # Ensure this text appears
        if "image" in script_name:
            img_path = "doc/img/output.jpg"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_tk = ctk.CTkImage(img.resize((self.result_frame.winfo_width(), self.result_frame.winfo_height()), Image.Resampling.LANCZOS))
                self.update_image_label(img_tk)
            else:
                self.result_label.configure(text=f"Result of {script_name}")  # Ensure this text appears
    def clear_angle_text(self):
        self.angle_text.configure(text="")
    def on_resize(self, event):
        if hasattr(self.image_label, 'image'):
            self.display_result(self.result_label.cget("text").split(" ")[-1])
    def on_closing(self):
        self.stop_all()
        os.kill(os.getpid(), signal.SIGTERM)
        self.destroy()
    def update_active_button(self):
        for button in self.button_widgets:
            if button.cget("text").startswith("Single-person") and self.active_module == "image_singlepose":
                button.configure(fg_color="#00FF00")
            elif button.cget("text").startswith("Multi-person Image") and self.active_module == "image_multipose":
                button.configure(fg_color="#00FF00")
            elif button.cget("text").startswith("Multi-person Video") and self.active_module == "video_multipose":
                button.configure(fg_color="#00FF00")
            elif button.cget("text").startswith("Multi-person Webcam") and self.active_module == "webcam_multipose":
                button.configure(fg_color="#00FF00")
            else:
                button.configure(fg_color="#FFFFFF")
if __name__ == "__main__":
    app = PoseEstimationApp()
    app.mainloop()
