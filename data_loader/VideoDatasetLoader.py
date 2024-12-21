import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

import os
import subprocess
from PIL import Image
import shutil
import sys

# Use this function to load the data path and label for the dataset
def data_and_words(path):
    words = os.listdir(path)
    words_list = []
    data = []
    for word in words:
        words_list.append(word)
        word_videos_path = os.path.join(path, word)
        videos = os.listdir(word_videos_path)
        for video in videos:
            data.append((os.path.join(word_videos_path, video), len(words_list)-1))

    words_dict = dict({})
    for num, word in enumerate(words_list):
        words_dict[word] = num

    return data, words_list, words_dict  # data will return a list of tuples where data[0] is the path to the video and data[1] is the label
                                         # words_list can be used to convert index to word
                                         # words_dict can be used to convert word to index




class VideoDatasetLoader(Dataset):
    def __init__(self, data, temp_data_folder, NUM_FRAMES=10, transform_frame=None,transform_video=None, video_fps=25, resolution='1920:1080'):
        self.data = data  # It should be a list of tuples where data[0] is the path to the video and data[1] is the label

        self.transform_frame = transform_frame  # Transformations to be done on the individual frames.
                                                # Recommended to use when transforms is required at frames level with some randomness, eg: Random Crop
                                                # Note: If the Dataset is showing tensor issue, try adding `ToTensor()` in transform.

        self.transform_video = transform_video  # Transformations to be done on the whole video.
                                                # Recommended to use when transforms is required at video level with some randomness, eg: Random Horizontal Flip

        self.NUM_FRAMES = NUM_FRAMES  # Number of frames to be extracted from the video

        self.fps = video_fps  # The fps at which the video will be saved by ffmpeg
                              # Note: Reducing this might give a small performance increase, which might add up when running it multiple times. But this will also lead to
                              # loss of some data, as some frames will be dropped by ffmpeg

        self.resolution = resolution  # resolution at which ffmpeg will save the frames (could be the same as the video or different).
                                      # Note: Reducing this might give a small performance increase, which might add up when running it multiple times.

        self.temp_data_folder = temp_data_folder  # A temporary folder where ffmpeg can store the frames of the video
                                                  # NOTE: this folder is recommended to be empty as frames get deleted from the folder after they are loaded as tensor!!


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

        video_images = os.listdir(output_video_path)

        # Setting variables for the range in for loop
        starting_frame = 0
        step_size = len(video_images)//self.NUM_FRAMES
        ending_frame = len(video_images)-len(video_images)%self.NUM_FRAMES  # Subtraction to remove the remainder so that we run the for loop extra

        # Selecting the NUM_FRAMES from the video
        for n in range(starting_frame, ending_frame, step_size):
            img = video_images[n]
            img_path = os.path.join(output_video_path, img)
            with Image.open(img_path) as pil_img:
                if self.transform_frame:
                    video_frames.append(self.transform_frame(pil_img))  # Note: Try adding ToTensor() in transform_frame, if any tensor related error arrises.
                    
                else:
                    video_frames.append(to_tensor(pil_img))

        try:
            video_frames = torch.stack(video_frames)  # Note: Try adding ToTensor() in transform_frame, if any tensor related error arrises.
        except TypeError:
            print(f"TypeError: Tried to stack {type(video_frames[0])}. Add ToTensor() in transform_frame!")
            sys.exit(1)
            return None, None
            


        if self.transform_video:
            video_frames = self.transform_video(video_frames)
        shutil.rmtree(output_video_path)

        label = self.data[idx][1]
        return video_frames, label