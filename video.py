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
# input_dir = os.path.join(root_p, 'input', 'video')
# output_dir = os.path.join(root_p, 'output', 'video')

video_dir = os.path.join(root_p, 'video')

if not os.path.exists(video_dir):
    os.makedirs(video_dir)


# video_folders = lambda s: sorted(
#    [name for name in os.listdir(s.input_dir) if os.path.isdir(os.path.join(s.input_dir, name))])
# turn the above in the a full function
def video_folders(s):
    ret = sorted(
        [name for name in os.listdir(s.input_dir) if os.path.isdir(os.path.join(s.input_dir, name))])
    return ret


def video_files(s):
    # any files in in_video_folders ending in .mp4
    ret = sorted([name for name in os.listdir(s.input_dir) if
                  os.path.isfile(os.path.join(s.input_dir, name)) and name.endswith('.mp4')])
    return ret


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

    input_dir = video_dir
    output_dir = video_dir

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
                "video_in": (video_files(cls),),
                "folder_out": (video_folders(cls),),
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
                "folder_in": (video_folders(cls),),
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
                "video_in": (video_files(cls),),
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
        import glob

        folder_in_path = os.path.join(self.input_dir, folder_in)
        image_list = []

        # List and sort all PNG or JPG files in the folder
        file_fps = sorted(
            glob.glob(os.path.join(folder_in_path, '*.png')) + glob.glob(os.path.join(folder_in_path, '*.jpg')))

        # Ensure the slice doesn't go beyond the end of the file list
        if idx_slice_stop + idx > len(file_fps):
            idx_slice_stop = len(file_fps) - idx

        for i in range(idx_slice_start + idx, idx_slice_stop + idx, slice_idx_step):
            for fn in file_fps[i:i + slice_idx_step]:
                image = cv2.imread(fn)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting to RGB format
                    image_list.append(image)

        # Convert list of images to tensor (T,H,W,C)
        tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in image_list])
        tensor = tensor / 255.0
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
            # image_list.append(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)

        # Convert list of images to tensor (T,H,W,C)
        tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in image_list])
        tensor = tensor / 255.0

        return (tensor,)


class ImageStackToVideoFrames(metaclass=WidgetMetaclass):
    """Really just saves the image stack to a folder"""

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "folder_out": ("STRING", {"multiline": False, "default": "video01"}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        image_stack = kwargs["image_stack"]
        folder_out = kwargs["folder_out"]
        folder_out_path = video_dir

        # full path to the output folder
        folder_out_path = os.path.join(video_dir, folder_out)

        # Create the output folder if it doesn't exist
        if not os.path.exists(folder_out_path):
            os.makedirs(folder_out_path)
        # move any current files in the output folder to a random folder inside the video folder
        import uuid
        import shutil
        random_folder = os.path.join(folder_out_path, str(uuid.uuid4()))
        os.makedirs(random_folder)
        for file in os.listdir(folder_out_path):
            file_path = os.path.join(folder_out_path, file)
            if os.path.isfile(file_path):
                shutil.move(file_path, random_folder)

        # Save each image in the image stack to the output folder
        for i in range(image_stack.shape[0]):
            image = image_stack[i].numpy()
            # make sure the image is in the correct color format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # convert to 0-255
            image = image * 255
            # use zfill to get a good ordering of the frames
            file_name = os.path.join(folder_out_path, "frame" + str(i).zfill(6) + ".jpg")
            # use the best compression nearly lossless
            cv2.imwrite(file_name, image, [cv2.IMWRITE_JPEG_QUALITY, 99])

        image_stack = None
        del image_stack
        return ()


class ImageStackToVideoFile(metaclass=WidgetMetaclass):
    """Really just saves the image stack to a video file"""

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "video_out": ("STRING", {"multiline": False, "default": "video01.mp4"}),
            },
            "optional":
                {"fps": ("FLOAT", {"default": 30, "min": 1, "max": 120}),
                 "func": ("FUNC", {"default": "NONE"}),
                 },
        }

    RETURN_TYPES = ("FUNC",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        """frames are in the format (T,H,W,C)"""
        import cv2
        import os
        import numpy as np
        import uuid
        import shutil
        import inspect

        func = kwargs.get("func", None)

        if func:
            kwargs = func(**kwargs)

        self.ARGS = kwargs
        self.IN_FUNC = func
        # get this functions source code
        self.CODE = inspect.getsource(self.handler)
        self.FUNC = self.handler

        try:
            image_stack = kwargs["image_stack"]
        except KeyError:
            # this means we are just passing our function through
            return (self,)

        video_out = kwargs["video_out"]
        fps = kwargs["fps"]
        video_out_path = os.path.join(video_dir, video_out)

        # check to see if the video file already exists
        if os.path.exists(video_out_path):
            # move the existing file to a random folder inside the video folder
            random_folder = os.path.join(video_dir, str(uuid.uuid4()))
            os.makedirs(random_folder)
            shutil.move(video_out_path, random_folder)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_out_path + ".mp4", fourcc, float(fps),
                                (image_stack.shape[2], image_stack.shape[1]))

        for i in range(image_stack.shape[0]):
            image = image_stack[i].numpy()

            # Check that the image is in the correct format
            if image.dtype != np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = (image * 255).astype(np.uint8)  # convert to 0-255 and cast to uint8

            video.write(image)

        video.release()
        image_stack = None
        del image_stack
        return (self,)


if __name__ == "__main__":
    print(NODE_CLASS_MAPPINGS)
