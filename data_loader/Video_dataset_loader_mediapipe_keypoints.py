import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# tag : v5.1

class VideoDataset_mp_kp(Dataset):
    def __init__(self, data, temp_data_folder, NUM_FRAMES=10, transform_frame=None, video_fps=25, resolution='1920:1080'):
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


    def landmark_to_list(self, landmarks):
        x, y =[], []
        for landmark in landmarks:
            x.append(landmark.x)
            y.append(landmark.y)
        return x, y


    def swap(self, left_thumb, right_thumb, hand):
        left = (left_thumb[0]-hand[0])**2 + (left_thumb[1]-hand[1])**2
        right = (right_thumb[0]-hand[0])**2 + (right_thumb[1]-hand[1])**2

        if right<left:
            return True
        return False


    def process_and_convert(self, image_path):

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
            
            if self.transform_frame:
                rgb_image = self.transform_frame(rgb_image)
            
            rgb_image = np.array(rgb_image)
            
            # Process pose landmarks
            pose_results = pose.process(rgb_image)
            x_pose, y_pose = self.landmark_to_list(pose_results.pose_landmarks.landmark[:24])
            

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


    def interpolate(self, arr):

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

        result = np.stack((arr_x, arr_y), axis=-1)
        return result
    
    def np_to_tensor(self, left_hand, right_hand, pose):
        landmark_list = []
        for n in range(self.NUM_FRAMES):
            x = np.concatenate((left_hand[0][n], pose[0][n], right_hand[0][n]))
            y = np.concatenate((left_hand[1][n], pose[1][n], right_hand[1][n]))
            result = np.stack((x,y))
            landmark_list.append(torch.from_numpy(result))
        landmark_tensor = torch.stack(landmark_list)
        landmark_tensor = landmark_tensor.clone().detach().to(torch.float32)
        return landmark_tensor

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

        video_images = os.listdir(output_video_path)

        # Setting variables for the range in for loop
        starting_frame = 0
        step_size = len(video_images)//self.NUM_FRAMES
        ending_frame = len(video_images)-len(video_images)%self.NUM_FRAMES  # Subtraction to remove the remainder so that we run the for loop extra

        # Selecting the NUM_FRAMES from the video
        for n in range(starting_frame, ending_frame, step_size):
            img = video_images[n]
            img_path = os.path.join(output_video_path, img)

            x_left_hand, y_left_hand, x_pose, y_pose, x_right_hand, y_right_hand = self.process_and_convert(img_path)
            left_hand[0].append(x_left_hand)
            left_hand[1].append(y_left_hand)
            right_hand[0].append(x_right_hand)
            right_hand[1].append(y_right_hand)
            pose[0].append(x_pose)
            pose[1].append(y_pose)

        left_hand= np.array(left_hand).transpose(1,2,0)
        right_hand = np.array(right_hand).transpose(1, 2, 0)
        pose = np.array(pose).transpose(1, 2, 0)

        left_hand = self.interpolate(left_hand)
        right_hand = self.interpolate(right_hand)
        pose = self.interpolate(pose)

        # Transpose back to the original shape
        left_hand = left_hand.transpose(2, 0, 1)
        right_hand = right_hand.transpose(2, 0, 1)
        pose = pose.transpose(2, 0, 1)
        tensor_landmarks = self.np_to_tensor(left_hand, right_hand, pose)
        label = self.data[idx][1]
        return torch.flatten(tensor_landmarks), label
