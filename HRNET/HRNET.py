import time
import cv2
import numpy as np
import onnxruntime
import math
from .utils import *
from .yolov6.YOLOv6 import YOLOv6

class HRNET:

    def __init__(self, path, model_type, conf_thres=0.7, search_region_ratio=0.1):
        self.conf_threshold = conf_thres
        self.model_type = model_type
        self.search_region_ratio = search_region_ratio

        # Initialize model
        self.initialize_model(path)

        # Map joint names to their index in the pose array
        self.joint_map = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

    def __call__(self, image, detections=None):
        if detections is None:
            return self.update(image)
        else:
            return self.update_with_detections(image, detections)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    # Update the path to the model file
    model_path = "hrnet_coco_w48_384x288.onnx"

    def update_with_detections(self, image, detections):
        full_height, full_width = image.shape[:2]

        if len(detections) == 0 or len(detections[0]) == 0:
            return None, None, None

        boxes, scores, class_ids = detections[0], detections[1], detections[2]

        poses = []
        angles = []
        total_heatmap = np.zeros((full_height, full_width))

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            box_width, box_height = x2 - x1, y2 - y1

            # Enlarge search region
            x1 = max(int(x1 - box_width * self.search_region_ratio), 0)
            x2 = min(int(x2 + box_width * self.search_region_ratio), full_width)
            y1 = max(int(y1 - box_height * self.search_region_ratio), 0)
            y2 = min(int(y2 + box_height * self.search_region_ratio), full_height)

            crop = image[y1:y2, x1:x2]
            body_heatmap, body_pose, extra_value = self.update(crop)  # Adjusted unpacking

            # Fix the body pose to the original image
            fixed_pose = body_pose + np.array([x1, y1])
            poses.append(fixed_pose)

            # Calculate angles for this pose
            pose_angles = self.calculate_angles(fixed_pose)
            angles.append(pose_angles)

            # Resize body_heatmap to match the region size if necessary
            body_heatmap_resized = cv2.resize(body_heatmap, (x2 - x1, y2 - y1))

            # Add the heatmap to the total heatmap
            total_heatmap[y1:y2, x1:x2] += body_heatmap_resized

        self.total_heatmap = total_heatmap
        self.poses = poses
        self.angles = angles

        return self.total_heatmap, self.poses, self.angles

    def update(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.total_heatmap, self.poses = self.process_output(outputs)

        # Calculate angles for single pose
        if self.poses is not None:
            self.angles = [self.calculate_angles(self.poses)]
        else:
            self.angles = None

        return self.total_heatmap, self.poses, self.angles

    def prepare_input(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, heatmaps):
        total_heatmap = cv2.resize(heatmaps.sum(axis=1)[0], (self.img_width, self.img_height))
        map_h, map_w = heatmaps.shape[2:]

        # Find the maximum value in each of the heatmaps and its location
        max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
        peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                        for heatmap in heatmaps[0, ...]])
        peaks[max_vals < self.conf_threshold] = np.array([np.nan, np.nan], dtype=np.float32)  # Ensure valid values for casting

        # Scale peaks to the image size
        peaks = peaks[:, ::-1] * np.array([self.img_width / map_w,
                                            self.img_height / map_h])

        return total_heatmap, peaks

    def calculate_angles(self, pose):
        angles = {}
        
        # Define the joint pairs for which we want to calculate angles (4 angles)
        joint_pairs = [
            ('left_shoulder', 'left_elbow', 'left_wrist', 'Shoulder angle left'),
            ('right_shoulder', 'right_elbow', 'right_wrist', 'Shoulder angle right'),
            ('left_elbow', 'left_shoulder', 'left_wrist', 'Elbow angle left'),
            ('right_elbow', 'right_shoulder', 'right_wrist', 'Elbow angle right'),
            ('left_shoulder', 'left_hip', 'left_knee', 'Hip angle left'),
            ('right_shoulder', 'right_hip', 'right_knee', 'Hip angle right'),
            ('left_hip', 'left_knee', 'left_ankle', 'Knee angle left'),
            ('right_hip', 'right_knee', 'right_ankle', 'Knee angle right')
        ]

        for joint1, joint2, joint3, angle_name in joint_pairs:
            p1 = pose[self.joint_map[joint1]]
            p2 = pose[self.joint_map[joint2]]
            p3 = pose[self.joint_map[joint3]]

            # Check if any of the joints are not detected (NaN)
            if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
                angles[angle_name] = None
            else:
                angle = self.calculate_angle(p1, p2, p3)
                angles[angle_name] = angle

        return angles

    def calculate_angle(self, p1, p2, p3):
        vector1 = p1 - p2
        vector2 = p3 - p2

        angle = np.arctan2(np.cross(vector1, vector2), np.dot(vector1, vector2))
        angle = np.abs(angle * 180.0 / np.pi)
        return angle

    def calculate_accuracy_per_point(self, detected_keypoints, ground_truth_keypoints, image_width, image_height):
        accuracies = {}
        max_distance = np.sqrt(image_width**2 + image_height**2)

        for joint_name, joint_index in self.joint_map.items():
            detected_point = detected_keypoints[joint_index]
            ground_truth_point = ground_truth_keypoints[joint_index]

            if not valid_point(detected_point) or not valid_point(ground_truth_point):
                accuracies[joint_name] = None
                continue

            distance = np.linalg.norm(detected_point - ground_truth_point)
            accuracy = 1 - (distance / max_distance)
            accuracies[joint_name] = accuracy

        return accuracies

    def draw_pose(self, image):
        if self.poses is None:
            return image
        return draw_skeletons(image, self.poses, self.model_type)

    def draw_heatmap(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        return draw_heatmap(image, self.total_heatmap, mask_alpha)

    def draw_all(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        img = self.draw_pose(self.draw_heatmap(image, mask_alpha))
        return self.draw_angles(img)

    def draw_angles(self, image):
        if self.poses is None or self.angles is None:
            return image

        for pose, angles in zip(self.poses, self.angles):
            for joint, angle in angles.items():
                if angle is not None:
                    joint_parts = joint.split(' ')
                    mid_joint = joint_parts[1]
                    x, y = pose[self.joint_map[mid_joint]]
                    cv2.putText(image, f"{angle:.1f}", (int(x) - 50, int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def draw_bounding_boxes(self, image, detections):
        boxes, scores, class_ids = detections
        if boxes is None or scores is None or class_ids is None:
            return image

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if box is None or len(box) != 4:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Person {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':
    from imread_from_url import imread_from_url

    # Initialize model
    model_path = "../models/hrnet_coco_w48_384x288.onnx"
    model_type = ModelType.COCO
    hrnet = HRNET(model_path, model_type, conf_thres=0.6)

    img = imread_from_url(
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/Bull-Riding2-Szmurlo.jpg")

    # Perform the inference in the image
    total_heatmap, poses, angles = hrnet(img)

    # Draw model output
    output_img = hrnet.draw_all(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_img)
    cv2.waitKey(0)