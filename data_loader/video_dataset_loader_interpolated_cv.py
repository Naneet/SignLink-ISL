import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import subprocess
import torchvision.transforms as transforms
import shutil
import mediapipe as mp
import cv2
import os


# tag : v5.2

class VideoDataset(Dataset):
    def __init__(self, data, temp_data_folder, NUM_FRAMES=10, transform_frame=None, video_fps=25, resolution='1920:1080', flip_prob=30, resize=300, crop=640):
        self.data = data  # It should be a list of tuples where data[0] is the path to the video and data[1] is the label

        self.transform_frame = transform_frame  # Transformations to be done on the individual frames.
                                                # Recommended to use when transforms is required at frames level with some randomness, eg: Random Crop

        self.NUM_FRAMES = NUM_FRAMES  # Number of frames to be extracted from the video

        self.fps = video_fps  # The fps at which the video will be saved by ffmpeg
                              # Note: Reducing this might give a small performance increase, which might add up when running it multiple times. But this will also lead to
                              # loss of some data, as some frames will be dropped by ffmpeg

        self.resolution = resolution  # resolution at which ffmpeg will save the frames (could be the same as the video or different).
                                      # Note: Reducing this might give a small performance increase, which might add up when running it multiple times.

        self.temp_data_folder = temp_data_folder  # A temporary folder where ffmpeg can store the frames of the video
                                                  # NOTE: this folder is recommended to be empty as frames get deleted from the folder after they are loaded as tensor!!

        self.flip_prob = flip_prob  # Probability of flipping a video

        self.frame_width = np.fromstring(resolution, sep=":")[0]  # For converting the coordinates from 0-1 to desired resolution
        self.frame_length = np.fromstring(resolution, sep=":")[1]

        self.rescale = transforms.Resize((resize,resize))  # For reducing the resolution of the image after plotting the landmarks
        self.crop = transforms.CenterCrop((crop,crop))     # Recommended to do in this way otherwise the model have difficulty generalising the dataset

    def landmark_to_list(self, landmarks):  # converting mediapipe landmarks to python list
        x, y =[], []
        for landmark in landmarks:
            x.append(landmark.x)
            y.append(landmark.y)
        return x, y


    def swap(self, left_thumb, right_thumb, hand):  # swapping hands side if it is assigned to wrong side
        left = (left_thumb[0]-hand[0])**2 + (left_thumb[1]-hand[1])**2
        right = (right_thumb[0]-hand[0])**2 + (right_thumb[1]-hand[1])**2

        if right<left:
            return True
        return False


    def process_and_convert(self, image_path, flip):

        # Initialize MediaPipe modules
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Load the image
        rgb_image = Image.open(image_path)

        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose, \
            mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:

            # Convert the BGR image to RGB
            if rgb_image.mode != 'RGB':
                rgb_image = rgb_image.convert('RGB')

            transform = transforms.RandomHorizontalFlip(p=1)
            if flip:
                rgb_image = transform(rgb_image)

            if self.transform_frame:
                rgb_image = self.transform_frame(rgb_image)

            rgb_image = np.array(rgb_image)

            # Process pose landmarks
            pose_results = pose.process(rgb_image)
            x_pose, y_pose = self.landmark_to_list(pose_results.pose_landmarks.landmark[:25])


            # Process hand landmarks
            hand_results = hands.process(rgb_image)
            if hand_results.multi_hand_landmarks:
                if len(hand_results.multi_hand_landmarks) > 0 and hand_results.multi_hand_landmarks[0] != None:
                    x_left_hand, y_left_hand = self.landmark_to_list(hand_results.multi_hand_landmarks[0].landmark)
                else:
                    x_left_hand, y_left_hand = [np.nan]*21, [np.nan]*21

                if len(hand_results.multi_hand_landmarks) > 1 and hand_results.multi_hand_landmarks[1] != None:
                    x_right_hand, y_right_hand = self.landmark_to_list(hand_results.multi_hand_landmarks[1].landmark)
                else:
                    x_right_hand, y_right_hand = [np.nan]*21, [np.nan]*21

                    if self.swap(left_thumb=(x_pose[21], y_pose[21]), right_thumb=(x_pose[22], y_pose[22]), hand=(x_left_hand[1], y_left_hand[1])):
                        x_left_hand, y_left_hand, x_right_hand, y_right_hand = x_right_hand, y_right_hand, x_left_hand, y_left_hand

            else:
                x_left_hand, y_left_hand = [np.nan]*21, [np.nan]*21
                x_right_hand, y_right_hand = [np.nan]*21, [np.nan]*21


            return x_left_hand, y_left_hand, x_pose, y_pose, x_right_hand, y_right_hand






    def interpolate(self, arr):  # interpolating missing landmarks

        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        if np.count_nonzero(~np.isnan(arr_x)) == 0:
            arr_x = np.zeros(arr_x.shape)
        if np.count_nonzero(~np.isnan(arr_y)) == 0:
            arr_y = np.zeros(arr_y.shape)

        arr_x = arr_x * self.frame_width
        arr_y = arr_y * self.frame_length

        result = np.stack((arr_x, arr_y), axis=-1)
        return result


    def align_hand_to_pose(self, hand_keypoints, wrist_pose, thumb_pose):
        """
        Aligns hand keypoints to pose wrist and thumb points, maintaining relative distances between keypoints.

        Parameters:
            hand_keypoints (np.ndarray): Array of shape (21, 2) representing hand keypoints.
            wrist_pose (np.ndarray): Array of shape (2,) representing pose wrist point.
            thumb_pose (np.ndarray): Array of shape (2,) representing pose thumb point.

        Returns:
            np.ndarray: New hand keypoints array of shape (21, 2) with adjusted positions.
        """
        # Create a new array to store the transformed hand keypoints
        aligned_hand_keypoints = np.copy(hand_keypoints)

        # Move WRIST to align with the pose wrist
        aligned_hand_keypoints[0] = wrist_pose

        # Calculate the vector from current WRIST to THUMB_TIP
        thumb_vector = hand_keypoints[2] - hand_keypoints[0]

        # Calculate the new vector from pose wrist to pose thumb
        new_thumb_vector = thumb_pose - wrist_pose

        # Calculate scaling factor and rotation
        scale_factor = np.linalg.norm(new_thumb_vector) / np.linalg.norm(thumb_vector)
        rotation_angle = np.arctan2(new_thumb_vector[1], new_thumb_vector[0]) - np.arctan2(thumb_vector[1], thumb_vector[0])
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle),  np.cos(rotation_angle)]
        ])

        # Maintain interdistance for all keypoints relative to WRIST
        for i in range(1, 21):
            # Calculate relative position of each keypoint
            relative_point = hand_keypoints[i] - hand_keypoints[0]

            # Scale and rotate the relative position
            new_point = rotation_matrix @ (relative_point * scale_factor)

            # Update the keypoint position in the new array
            aligned_hand_keypoints[i] = aligned_hand_keypoints[0] + new_point

        return aligned_hand_keypoints

    

    def plot_hand(self, hand, img):
        for coord in hand:
            x, y = coord[0], coord[1]
            cv2.circle(img, (int(x), int(y)), radius=5, color=(0,255,0), thickness=-1)
        return img


    def plot_pose(self, pose, img):
        for coord in pose:
            x, y = coord[0], coord[1]
            cv2.circle(img, (int(x), int(y)), radius=5, color=(0,255,0), thickness=-1)
        return img

    def draw_connections(self, img, landmarks, connections, color=(255, 255, 255)):
      for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if not (np.isnan(start).any() or np.isnan(end).any()):
                cv2.line(img, (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])), color=color, thickness=2)
      return img

    def plot_landmark(self, left_hand, pose, right_hand):
        mylist = []
        for n in range(self.NUM_FRAMES):
            width, height = map(int, self.resolution.split(":"))
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image = self.plot_hand(hand=left_hand[n], img=image)
            image = self.plot_hand(hand=right_hand[n], img=image)
            image = self.plot_pose(pose=pose[n], img=image)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pose_connections = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (18, 20),
                (11, 23), (12, 24), (23, 24), (15, 21), (16, 22)
            ]

            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20) # Pinky finger
            ]
            rgb_image = self.draw_connections(image, pose[n], pose_connections)
            rgb_image = self.draw_connections(image, left_hand[n], hand_connections)
            rgb_image = self.draw_connections(image, right_hand[n], hand_connections)
            # plt.figure(figsize=(12,8))
            # plt.imshow(rgb_image)
            # plt.axis("off")
            # plt.show()
            mylist.append(self.rescale(self.crop(torch.permute(torch.from_numpy(rgb_image).to(torch.uint8),(2,0,1)))))
        return torch.stack(mylist)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        video_path = self.data[idx][0]
        video_file_name = os.path.basename(video_path)  # Assigns the name of the video to the variable
        output_video_path = os.path.join(self.temp_data_folder, video_file_name)  # creating a temp path in the temp_folder for saaving the frames of the video
        os.makedirs(output_video_path, exist_ok=True)  # Creating a folder with the name of the video in the temp_data_folder to save the frames of the video

        # Command to convert the video to frames
        fallback_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', f'fps={self.fps},scale={self.resolution},format=yuv420p',
                    '-q:v', '2',
                    os.path.join(output_video_path, 'img_%05d.jpg')
        ]
        subprocess.run(fallback_cmd, check=True, stderr=subprocess.PIPE)

        video_frames = []
        left_hand, right_hand, pose = [[],[]], [[],[]], [[],[]]

        if np.random.randint(low=1, high=100, size=None, dtype=int) <= self.flip_prob:
            flip=True
        else:
            flip=False

        video_images = os.listdir(output_video_path)

        # Setting variables for the range in for loop
        starting_frame = 0
        step_size = len(video_images)//self.NUM_FRAMES
        ending_frame = len(video_images)-len(video_images)%self.NUM_FRAMES  # Subtraction to remove the remainder so that we run the for loop extra

        # Selecting the NUM_FRAMES from the video
        for n in range(starting_frame, ending_frame, step_size):
            img = video_images[n]
            img_path = os.path.join(output_video_path, img)

            x_left_hand, y_left_hand, x_pose, y_pose, x_right_hand, y_right_hand = self.process_and_convert(img_path, flip)
            left_hand[0].append(x_left_hand)
            left_hand[1].append(y_left_hand)
            right_hand[0].append(x_right_hand)
            right_hand[1].append(y_right_hand)
            pose[0].append(x_pose)
            pose[1].append(y_pose)

        left_hand= np.array(left_hand).transpose(1,2,0)
        right_hand = np.array(right_hand).transpose(1, 2, 0)
        pose = np.array(pose).transpose(1, 2, 0)

        pose = self.interpolate(pose)
        left_hand = self.interpolate(left_hand, )#pose[::,22])
        right_hand = self.interpolate(right_hand, )#pose[::,21])

        vid_tensor = self.plot_landmark(left_hand=left_hand, pose=pose, right_hand=right_hand)
        shutil.rmtree(output_video_path)
        label = self.data[idx][1]
        return vid_tensor, label
