### info for code completion AI ###
"""
all of these classes are plugins for comfyui and follow the same pattern
all of the images are torch tensors and it is unknown and unimportant if they are on the cpu or gpu
all image inputs are (B,W,H,C)

avoid numpy and PIL as much as possible
"""
import os
from abc import ABC, ABCMeta, abstractmethod
from typing import List, Tuple, Union
import cv2
import torch
from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture
from torch import dtype

NODE_CLASS_MAPPINGS = {}  # this is the dictionary that will be used to register the nodes

# setup directories
root_p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
input_dir = os.path.join(root_p, 'input', 'video')
output_dir = os.path.join(root_p, 'output', 'video')

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

out_video_folders = lambda s: sorted(
    [name for name in os.listdir(s.output_dir) if os.path.isdir(os.path.join(s.output_dir, name))])

in_video_folders = lambda s: sorted(
    [name for name in os.listdir(s.input_dir) if os.path.isdir(os.path.join(s.input_dir, name))])

# any files in in_video_folders ending in .mp4
in_video_files = lambda s: sorted(
    [name for name in os.listdir(s.input_dir) if
     os.path.isfile(os.path.join(s.input_dir, name)) and name.endswith('.mp4')])


# start with an ABC to define the common widget interface

class WidgetMetaclass(ABCMeta):
    """A metaclass that automatically registers classes."""

    def __init__(cls, name, bases, attrs):
        if (ABC in bases) or (VideoWidget in bases):
            pass
        else:
            NODE_CLASS_MAPPINGS[name] = cls

        super().__init__(name, bases, attrs)


class VideoWidget(ABC, metaclass=WidgetMetaclass):
    """Abstract base class for simple construction"""

    input_dir = input_dir
    output_dir = output_dir

    CATEGORY = "video"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "handler"

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls):
        """All subclasses must provide a INPUT_TYPES classmethod"""
        pass

    @abstractmethod
    def handler(self, one: object, two: object):
        """All subclasses must provide a handler method"""
        pass


class VideoFileToVideoFolder(VideoWidget, metaclass=WidgetMetaclass):
    """Abstract base class for simple construction"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "video_in": (in_video_files(cls),),
                "folder_out": (out_video_folders(cls),),
            },

            "optional":
                {"text":
                     ("STRING", {"multiline": False}),
                 }
        }

    @abstractmethod
    def handler(self, one, two, text):
        return (one, two,)

class VideoFolderToImage(VideoWidget, metaclass=WidgetMetaclass):
    """Abstract base class for simple construction"""
    RETURN_TYPES = ("IMAGE",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "folder_in": (in_video_folders(cls),),
                "idx_slice_start": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 0}),
                "idx_slice_stop": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 1}),
                "slice_idx_step": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 1}),
                "idx": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 0}),
            },

            "optional":
                {"text":
                     ("STRING", {"multiline": False}),
                 }
        }

    @abstractmethod
    def handler(self, one, two, text):
        return (one,)

class VideoFileToImage(VideoWidget, metaclass=WidgetMetaclass):
    """Abstract base class for simple construction"""
    RETURN_TYPES = ("IMAGE",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "video_in": (in_video_files(cls),),
                "idx_slice_start": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 0}),
                "idx_slice_stop": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 1}),
                "slice_idx_step": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 1}),
                "idx": ("INT", {"min": 0, "max": 10000, "step": 1, "default": 0}),
            },

            "optional":
                {"text":
                     ("STRING", {"multiline": False}),
                 }
        }

    @abstractmethod
    def handler(self, one, two, text):
        return (one,)

## end of abstract base classes ##

class VideoToFrames(VideoFileToVideoFolder, metaclass=WidgetMetaclass):
    """use cv2 to open the video and use it as a source for image stacks"""

    RETURN_TYPES = ("IMAGE",)

    def __init__(self):
        super().__init__()
        self.old = self.INPUT_TYPES
        # self.INPUT_TYPES = lambda s: self.old().update({})

    def handler(self, video_in, folder_out, text):
        video_in_path = os.path.join(self.input_dir, video_in)
        video_out_path = os.path.join(self.output_dir, folder_out)

        vidcap = cv2.VideoCapture(video_in_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_out_path, "frame%d.jpg" % count), image)
            success, image = vidcap.read()
            count += 1

        return (folder_out,)


class VideoFramesToImageStack(VideoFolderToImage, metaclass=WidgetMetaclass):
    """Use cv2 to open the video and save the videos individual frames"""

    def handler(self, folder_in, text, idx_slice_start, idx_slice_stop, slice_idx_step, idx):
        """
        Return slices of time from the video, sequences of images, defined by the start, stop and step
        - torch tensors (T,H,W,C)
        """
        folder_in_path = os.path.join(self.input_dir, folder_in)
        image_list = []

        for i in range(idx_slice_start+idx, idx_slice_stop+idx, slice_idx_step):
            file_fps = [os.path.join(folder_in_path, f"frame{i+ii}.jpg") for ii in range(slice_idx_step)]
            for fn in file_fps:
                image = cv2.imread(fn)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting to RGB format
                    image_list.append(image)

        # Convert list of images to tensor (T,H,W,C)
        tensor = torch.stack([torch.tensor(img,dtype=torch.float32) for img in image_list])
        tensor=tensor/255.0


        return (tensor,)


class VideoFileToImageStack(VideoFileToImage, metaclass=WidgetMetaclass):
    """Use cv2 to open a video and save the videos individual frames"""

    def handler(self, video_in, text, idx_slice_start, idx_slice_stop, slice_idx_step, idx):
        """
        Return slices of time from the video, sequences of images, defined by the start, stop and step
        - torch tensors (T,H,W,C)
        """
        image_list = []
        video_in_fp = os.path.join(self.input_dir, video_in)
        vidcap = cv2.VideoCapture(video_in_fp)
        if not vidcap.isOpened():
            print("Error: Unable to open video file")
            return


        start = (idx * slice_idx_step) + idx_slice_start
        stop = (idx * slice_idx_step) + idx_slice_stop

        for frame_count in range(start, stop, slice_idx_step):
            # Seek to frame_count frame
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            success, image = vidcap.read()
            if not success:
                print("Error: Unable to read frame from video file")
                break
            #image_list.append(image)


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)

        # Convert list of images to tensor (T,H,W,C)
        tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in image_list])
        tensor = tensor / 255.0

        return (tensor,)


if __name__ == "__main__":
    print(NODE_CLASS_MAPPINGS)
