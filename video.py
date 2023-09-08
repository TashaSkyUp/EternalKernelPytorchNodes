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
from custom_nodes.EternalKernelLiteGraphNodes.image import torch_image_show
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
import imageio_ffmpeg as ffmpeg
import os
import torch
from torchvision.utils import draw_keypoints
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import cv2
import subprocess
import imageio_ffmpeg as ffmpeg

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


def run_ffmpeg_command(args):
    """
    Executes a command using the ffmpeg executable.

    Parameters:
        args (list): List of command-line arguments for ffmpeg.
    """
    cmd = [ffmpeg.get_ffmpeg_exe()] + args
    subprocess.run(cmd)


# start with an ABC to define the common widget interface

class ABCWidgetMetaclass(ABCMeta):
    """A metaclass that automatically registers classes."""

    def __init__(cls, name, bases, attrs):
        if (ABC in bases) or (ABCVideoWidget in bases) or ("ABC" in name):
            pass
        else:
            NODE_CLASS_MAPPINGS[name] = cls

        super().__init__(name, bases, attrs)


class ABCVideoWidget(ABC, metaclass=ABCWidgetMetaclass):
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


class ABCVideoFileToABCVideoFolder(ABCVideoWidget, metaclass=ABCWidgetMetaclass):
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


class ABCABCVideoFolderToImage(ABCVideoWidget, metaclass=ABCWidgetMetaclass):
    """Abstract base class for simple construction"""
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "folder_in": ("STRING", {"default": "full_path_to_folder"}),
                "idx_slice_start": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0}),
                "idx_slice_stop": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 1}),
                "slice_idx_step": ("INT", {"min": 0, "max": 9999, "step": 1, "default": 1}),
                "idx": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0}),
            },

            "optional":
                {"text":
                     ("STRING", {"multiline": False}),
                 }
        }

    @abstractmethod
    def handler(self, one, two, text):
        return (one,)


class ABCABCVideoFileToImage(ABCVideoWidget, metaclass=ABCWidgetMetaclass):
    """Abstract base class for simple construction"""
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "video_in": ("STRING", {"default": "full_path_to_video"}),
                "idx_slice_start": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0}),
                "idx_slice_stop": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 1}),
                "slice_idx_step": ("INT", {"min": 0, "max": 9999, "step": 1, "default": 1}),
                "idx": ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0}),
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

class VideoToFramesFolder(ABCVideoFileToABCVideoFolder, metaclass=ABCWidgetMetaclass):
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


class VideoFramesFolderToImageStack(metaclass=ABCWidgetMetaclass):
    """Use cv2 to open the video and save the videos individual frames"""

    @classmethod
    def INPUT_TYPES(cls):
        from addict import Dict
        ret = Dict()
        ret.required = Dict()
        ret.required.folder_in = ("STRING", {"default": "full_path_to_folder"})
        ret.required.use_subfolders = ([True, False], {"default": False})
        ret.required.by_date_or_name = (["DATE", "NAME"], {"default": "DATE"})
        ret.required.idx_slice_start = ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0})
        ret.required.idx_slice_stop = ("INT", {"min": 0, "max": 99999, "step": 1, "default": 1})
        ret.required.slice_idx_step = ("INT", {"min": 0, "max": 9999, "step": 1, "default": 1})
        ret.required.idx = ("INT", {"min": 0, "max": 99999, "step": 1, "default": 0})
        ret.required.use_float16 = ([True, False], {"default": False})

        return ret

    CATEGORY = "video"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "handler"

    @torch.inference_mode()
    def handler(self, folder_in,
                use_subfolders,
                by_date_or_name,
                idx_slice_start,
                idx_slice_stop,
                slice_idx_step,
                idx,
                use_float16
                ):
        """
        Return slices of time from the video, sequences of images, defined by the start, stop and step
        - torch tensors (T,H,W,C)
        """

        import glob
        idx = idx * slice_idx_step

        folder_in_path = folder_in
        image_list = []

        p = os.path.abspath(folder_in_path)
        # List and sort all PNG or JPG files in the folder
        if not use_subfolders and by_date_or_name == "NAME":
            file_fps = sorted(
                glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.jpg')))
        elif not use_subfolders and by_date_or_name == "DATE":
            file_fps = sorted(
                glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.jpg')),
                key=os.path.getmtime)
        elif use_subfolders and by_date_or_name == "NAME":
            file_fps = sorted(
                glob.glob(os.path.join(p, '**/*.png'), recursive=True) +
                glob.glob(os.path.join(p, '**/*.jpg'), recursive=True))
        elif use_subfolders and by_date_or_name == "DATE":
            file_fps = sorted(
                glob.glob(os.path.join(p, '**/*.png'), recursive=True) +
                glob.glob(os.path.join(p, '**/*.jpg'), recursive=True),
                key=os.path.getmtime)

        # Ensure the slice doesn't go beyond the end of the file list

        if idx_slice_stop + idx > len(file_fps):
            idx_slice_stop = len(file_fps) - idx

        slice_size = idx_slice_stop - idx_slice_start
        # for i in range(idx_slice_start + idx, idx_slice_stop + idx):
        for fn in file_fps[idx_slice_start + idx:idx_slice_start + idx + slice_size]:
            image = cv2.imread(fn)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting to RGB format
                image_list.append(image)

        # Convert list of images to tensor (T,H,W,C)
        if use_float16:
            tensor = torch.stack([torch.tensor(img, dtype=torch.float16) for img in image_list])
        else:
            tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in image_list])
        del image_list
        tensor = tensor / 255.0
        return (tensor,)


class SmoothStackTemporalByDistance2(ABCABCVideoFolderToImage, metaclass=ABCWidgetMetaclass):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "top_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.1, "default": 1.0}),
                "bottom_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.1, "default": 1.0}),
                "max_iterations": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 1}),
                "target_deviation": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 0.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")

    def handler(self, image_stack, top_thresh_normed=0.9, bottom_thresh_normed=0.10, max_iterations=1,
                target_deviation=0.0):
        import torch
        # Code for printing memory size and available GPU memory stays the same

        image_stack = image_stack.to(torch.float16)
        info = ""
        iteration = 0
        while True:
            # Compute the first-order differences (gradient) along the time dimension (0)
            first_order_diff = torch.gradient(image_stack, dim=0)[0]

            # Compute the Euclidean norm of the first-order differences
            euclidean_diff_first_order = torch.norm(first_order_diff.view(first_order_diff.shape[0], -1), dim=1).clone()

            euclidean_diff_first_order_range = euclidean_diff_first_order.max() - euclidean_diff_first_order.min()
            euclidean_diff_first_order_normed = (euclidean_diff_first_order - euclidean_diff_first_order.min()) / \
                                                euclidean_diff_first_order_range

            euclidean_diff_first_order_normed = euclidean_diff_first_order_normed[:-1]

            start_mean = euclidean_diff_first_order.mean()

            start_std = euclidean_diff_first_order.std()

            if start_std <= target_deviation or iteration >= max_iterations:
                break

            # Identify the frames with differences greater than the top threshold, excluding the last index
            outlier_indices_top = torch.where(euclidean_diff_first_order_normed > top_thresh_normed)[0]

            # Interpolate frames where the differences are greater than the top threshold
            interpolated_frames_top = 0.5 * (image_stack[outlier_indices_top] + image_stack[outlier_indices_top + 1])
            image_stack[outlier_indices_top + 1] = interpolated_frames_top

            # Identify the frames with differences less than the bottom threshold, excluding the last index
            outlier_indices_bottom = torch.where(euclidean_diff_first_order_normed < bottom_thresh_normed)[0]

            # Create a mask to delete frames where the differences are less than the bottom threshold
            mask = torch.ones(image_stack.size(0), dtype=torch.bool)
            mask[outlier_indices_bottom + 1] = False
            image_stack = image_stack[mask]
            # now remove the first and last frames
            # image_stack = image_stack[1:-1]
            # print the shape and stats for image_stack

            # relate how many frames where changed via top threshold and bottom threshold
            # also report on the mean and std of the euclidean_diff_first_order

            info += f"Number of frames changed via top threshold: {len(outlier_indices_top)}\n " \
                    f"Number of frames changed via bottom threshold: {len(outlier_indices_bottom)}\n " \
                    f"Mean of euclidean_diff_first_order: {start_mean}\n " \
                    f"Std of euclidean_diff_first_order: {start_std}\n " \
                    f"\n\n"
            print(info)
            iteration += 1

            if image_stack.shape[0] == 0:
                print("image_stack is empty, returning None")
                return (None, None,)
            print(
                f"image_stack shape: {image_stack.shape}, image_stack stats: {image_stack.min()}, {image_stack.max()}")

        return (image_stack.to(torch.float32), info,)


class SmoothStackTemporalByDistance(ABCABCVideoFolderToImage, metaclass=ABCWidgetMetaclass):
    """
    Smooth the temporal dimension of the image stack by calculating the distance between each frame
    and the next frame and then averaging the frames that are within a certain distance of each other.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "top_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.1, "default": 1.0}),
                "bottom_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.1, "default": 1.0}),
                "iterations": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")

    def handler(self, image_stack,
                top_thresh_normed=0.9, bottom_thresh_normed=0.10, iterations=1):
        import torch
        # calculate and print the total size in memory in bytes of the tensor
        print(
            f"Size of tensor in memory in GB: {image_stack.element_size() * image_stack.nelement() / (1024 ** 3)} GB, dtype: {image_stack.dtype}")
        # print the available memory on the GPU in GB
        print(f"Available memory on GPU in GB: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)} GB")

        # image_stack = image_stack.clone()
        image_stack = image_stack.to(torch.float16)
        info = ""
        for _ in range(iterations):
            # Compute the first-order differences (gradient) along the time dimension (0)
            first_order_diff = torch.gradient(image_stack, dim=0)[0]

            # Compute the Euclidean norm of the first-order differences
            euclidean_diff_first_order = torch.norm(first_order_diff.view(first_order_diff.shape[0], -1), dim=1).clone()

            euclidean_diff_first_order_range = euclidean_diff_first_order.max() - euclidean_diff_first_order.min()
            euclidean_diff_first_order_normed = (euclidean_diff_first_order - euclidean_diff_first_order.min()) / \
                                                euclidean_diff_first_order_range

            euclidean_diff_first_order_normed = euclidean_diff_first_order_normed[:-1]

            start_mean = euclidean_diff_first_order.mean()
            start_std = euclidean_diff_first_order.std()

            # Identify the frames with differences greater than the top threshold, excluding the last index
            outlier_indices_top = torch.where(euclidean_diff_first_order_normed > top_thresh_normed)[0]

            # Interpolate frames where the differences are greater than the top threshold
            interpolated_frames_top = 0.5 * (image_stack[outlier_indices_top] + image_stack[outlier_indices_top + 1])
            image_stack[outlier_indices_top + 1] = interpolated_frames_top

            # Identify the frames with differences less than the bottom threshold, excluding the last index
            outlier_indices_bottom = torch.where(euclidean_diff_first_order_normed < bottom_thresh_normed)[0]

            # Create a mask to delete frames where the differences are less than the bottom threshold
            mask = torch.ones(image_stack.size(0), dtype=torch.bool)
            mask[outlier_indices_bottom + 1] = False
            image_stack = image_stack[mask]
            # now remove the first and last frames
            # image_stack = image_stack[1:-1]
            # print the shape and stats for image_stack

            # relate how many frames where changed via top threshold and bottom threshold
            # also report on the mean and std of the euclidean_diff_first_order
            info += f"Number of frames changed via top threshold: {len(outlier_indices_top)}\n " \
                    f"Number of frames changed via bottom threshold: {len(outlier_indices_bottom)}\n " \
                    f"Mean of euclidean_diff_first_order: {start_mean}\n " \
                    f"Std of euclidean_diff_first_order: {start_std}\n " \
                    f"\n\n"
            print(info)

            if image_stack.shape[0] == 0:
                print("image_stack is empty, returning None")
                return (None, None,)
            print(
                f"image_stack shape: {image_stack.shape}, image_stack stats: {image_stack.min()}, {image_stack.max()}")

        return (image_stack.to(torch.float32), info,)


class GetImageStackStatistics(ABCABCVideoFolderToImage, metaclass=ABCWidgetMetaclass):
    """Get statistics of an image stack"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "method": (["SSIM", "MSE"], {"default": "SSIM"}),
            },

        }

    RETURN_TYPES = ("STRING", "IMAGE",)

    def handler(self, image_stack, method):
        import numpy as np
        import torch
        # if the dtype is not float32
        if image_stack.dtype != torch.float32:
            image_stack = image_stack.to(torch.float32)

        image_stack_BCWH = image_stack.permute(0, 3, 1, 2)
        image_stack_BCWH_DIFF = image_stack_BCWH[1:] - image_stack_BCWH[:-1]

        # we want to calculate the mean squared error
        # of the image stack
        # so we need to calculate the difference between each image in the stack and the previous image
        # we have that in image_stack_BCWH_DIFF
        # now we need to calculate the mean squared error of each image in the stack
        # which will give us a tensor of shape (B)
        # where B is the number of images in the stack

        if method == "MSE":
            stack_mse_B = torch.mean(image_stack_BCWH_DIFF ** 2, dim=(1, 2, 3))
        else:
            # not implimented
            raise NotImplementedError

        image_stack_diff_len = len(stack_mse_B)
        required_image_info_width = 512 + image_stack_diff_len

        mean = torch.mean(stack_mse_B)
        std = torch.std(stack_mse_B)
        median = torch.median(stack_mse_B)
        max = torch.max(stack_mse_B)
        min = torch.min(stack_mse_B)

        out_str = f"mean: {mean}\n std: {std}\n median: {median}\n max: {max}\n min: {min}\n len: {image_stack_diff_len}"

        # create the histogram
        hist_values, _ = np.histogram(stack_mse_B.numpy(), bins=100)
        hist_values = hist_values / hist_values.max()  # Normalize to [0, 1]

        # Create an image tensor with shape (1, 512, required_width, 3) filled with zeros
        out_hist_image = torch.zeros((1, 512, required_image_info_width, 3))

        # Determine the scaling factors for width and height for the histogram
        scale_width = 512 // len(hist_values)
        scale_height = out_hist_image.shape[1]

        # Fill the image tensor with the histogram
        for i, value in enumerate(hist_values):
            height = int(value * scale_height)
            start_x = int(i * scale_width)
            end_x = int((i + 1) * scale_width)
            out_hist_image[0, height:, start_x:end_x - 1, :] = 1  # Color as white

        # Create a plot for the difference for each frame pair
        plot_offset = 512
        for i, value in enumerate(stack_mse_B):
            height = int(value * scale_height / max)  # Normalize by the max value
            start_x = plot_offset + i
            end_x = plot_offset + i + 1
            out_hist_image[0, height:, start_x:end_x, :] = 1  # Color as white

        return (out_str, out_hist_image,)


class VideoFileToImageStack(ABCABCVideoFileToImage, metaclass=ABCWidgetMetaclass):
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
        slice_length = stop - start

        for frame_count in range(start, start + slice_length):
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


class ImageStackToVideoFramesFolder(metaclass=ABCWidgetMetaclass):
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
        import tempfile
        import uuid
        import shutil

        image_stack = kwargs["image_stack"]
        folder_out = kwargs["folder_out"]
        folder_out_path = video_dir

        # full path to the output folder
        folder_out_path = os.path.join(video_dir, folder_out)

        # Create the output folder if it doesn't exist
        if not os.path.exists(folder_out_path):
            os.makedirs(folder_out_path)

        # move any current files in the output folder to a random folder inside the video folder
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


class TemporalSpatialSmoothing(metaclass=ABCWidgetMetaclass):
    """Smooths the image stack in time and space"""

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "spatial_kernel_size": ("INT", {"default": 3, "min": 1, "max": 50}),
                "temporal_kernel_size": ("INT", {"default": 3, "min": 1, "max": 50}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    @staticmethod
    def gaussian_kernel_1d(kernel_size=3, sigma=1.0):
        import numpy as np
        if kernel_size % 2 == 0:
            kernel_size += 1
        x_coord = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).float()
        gaussian_kernel = (1. / (sigma * torch.sqrt(torch.tensor(2. * np.pi)))) * torch.exp(
            -x_coord ** 2. / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel

    def handler(self, **kwargs):
        import numpy as np
        import torch
        import torch.nn as nn

        image_stack = kwargs["image_stack"]
        spatial_kernel_size = kwargs["spatial_kernel_size"]
        temporal_kernel_size = kwargs["temporal_kernel_size"]
        spatial_sigma = kwargs.get("spatial_sigma", 1.0)
        temporal_sigma = kwargs.get("temporal_sigma", 1.0)

        # make sure image_stack is a torch tensor and has the right dimensions
        if not torch.is_tensor(image_stack):
            image_stack = torch.tensor(image_stack, dtype=torch.float32)

        if len(image_stack.shape) < 5:  # if it doesn't have a channel dimension
            image_stack = image_stack.unsqueeze(0)  # add a batch dimension
            image_stack = image_stack.permute(0, 4, 1, 2,
                                              3)  # rearrange dimensions to: (batch, channel, depth, height, width)

        # create 1D Gaussian kernels for each dimension
        spatial_kernel = self.gaussian_kernel_1d(spatial_kernel_size, spatial_sigma)
        temporal_kernel = self.gaussian_kernel_1d(temporal_kernel_size, temporal_sigma)

        # apply the convolution along each dimension separately
        for dim, kernel in enumerate([temporal_kernel, spatial_kernel, spatial_kernel]):
            padding_size = kernel.shape[0] // 2
            # define the padding differently for each dimension
            if dim == 0:  # depth (temporal)
                padding = nn.ReplicationPad3d((0, 0, 0, 0, padding_size, padding_size))
            elif dim == 1:  # height
                padding = nn.ReplicationPad3d((0, 0, padding_size, padding_size, 0, 0))
            elif dim == 2:  # width
                padding = nn.ReplicationPad3d((padding_size, padding_size, 0, 0, 0, 0))
            image_stack = padding(image_stack)
            if dim == 0:  # depth/temporal
                conv = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(kernel.shape[0], 1, 1), groups=3,
                                 bias=False)
                reshaped_kernel = kernel[None, None, :, None, None]
            elif dim == 1:  # height
                conv = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, kernel.shape[0], 1), groups=3,
                                 bias=False)
                reshaped_kernel = kernel[None, None, None, :, None]
            elif dim == 2:  # width
                conv = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, 1, kernel.shape[0]), groups=3,
                                 bias=False)
                reshaped_kernel = kernel[None, None, None, None, :]
            conv.weight = nn.Parameter(reshaped_kernel.repeat(3, 1, 1, 1, 1), requires_grad=False)
            image_stack = conv(image_stack)

        # remove the batch dimension and rearrange dimensions back to: (depth, height, width, channel)
        image_stack = image_stack.squeeze(0).permute(1, 2, 3, 0)

        return (image_stack,)


class TemporalOpticalFlowSmoothing(metaclass=ABCWidgetMetaclass):
    """Smooths the image stack in time using RAFT for optical flow estimation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "k": ("INT", {"default": 3, "min": 1, "max": 120}),
                "device": ("STRING", ["cpu", "cuda"]),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    @staticmethod
    def make_grid(flow):
        # Assumes flow is in shape Bx2xHxW
        B, _, H, W = flow.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H))
        grid = torch.stack((grid_y.t(), grid_x.t()), 2).unsqueeze(0)
        grid = grid.repeat(B, 1, 1, 1)  # Make it BxHxWx2
        grid = grid.permute(0, 3, 1, 2)  # Convert to Bx2xHxW
        grid = grid.type_as(flow)  # Make sure the data type is the same
        return grid + flow

    @staticmethod
    def worker(args):
        import torch.nn.functional as F
        k, image_stack, device = args
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        model = model.eval()

        img1 = image_stack[0].unsqueeze(0)
        img2 = image_stack[1].unsqueeze(0)

        # Remove batch dimension if present
        while len(img1.shape) > 4:
            print(f"Batch dimension detected. for image shape {img1.shape}")
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)

        # Permute tensor dimensions to match expected input for model
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)

        # Move images to the specified device
        img1, img2 = img1.to(device), img2.to(device)

        flow = model(img1, img2)[-1]

        # Create intermediary frames
        frames = []
        for i in range(k):
            factor = i / k
            # factor = factor * 2 - 1

            # normalize flow
            n_flow = flow.clone()
            print(n_flow.min(), n_flow.max(), n_flow.dtype)
            # n_flow[:,0,:,:] = n_flow[:,0,:,:] / (n_flow[:,0,:,:].max() - n_flow[:,0,:,:].min())
            # n_flow[:,1,:,:] = n_flow[:,1,:,:] / (n_flow[:,1,:,:].max() - n_flow[:,1,:,:].min())
            # print(n_flow.min(), n_flow.max(), n_flow.dtype)
            # n_flow[:, 0, :, :] = n_flow[:, 0, :, :] - n_flow[:, 0, :, :].min()
            # n_flow[:, 1, :, :] = n_flow[:, 1, :, :] - n_flow[:, 1, :, :].min()
            # print (n_flow.min(), n_flow.max(),n_flow.dtype)
            n_flow = n_flow / n_flow.shape[2]
            print(n_flow.min(), n_flow.max(), n_flow.dtype)

            flow_interpolated = n_flow * factor
            grid = TemporalOpticalFlowSmoothing.make_grid(flow_interpolated)
            grid = grid.permute(0, 2, 3, 1)
            warped_image = F.grid_sample(img1, grid, mode='bicubic', padding_mode='zeros')
            warped_image = warped_image.permute(0, 2, 3, 1).detach().cpu()  # Convert it back to BxHxWxC
            frames.append(warped_image)

        return frames

    @staticmethod
    def calculate_optical_flow(image_stack, device, k):
        # from torch.multiprocessing import Pool, set_start_method

        # set_start_method('spawn')  # set the start method to spawn
        # with Pool(2) as pool:
        #    # create the tasks in an intellegent way, detaching each stack, with each stack being only the frames needed
        #    tasks = []
        #    for t in range(image_stack.shape[0] - 1):
        #        needed_frames = [t, t + 1]  # the frames needed for this task
        #        stack = image_stack[needed_frames, :, :, :]  # the stack of frames needed for this task
        #        stack = stack.clone()
        #        tasks.append((2, stack.detach().cpu(), device))  # add the task to the list of tasks
        #    flows = pool.map(TemporalOpticalFlowSmoothing.worker, tasks)
        #    # print(f"flows: {flows}")
        #    # print(f"flows[0]: {flows[0]}")

        ##map(TemporalOpticalFlowSmoothing.worker, tasks)

        flows = []
        for i in range(image_stack.shape[0] - 1):
            needed_frames = [i, i + 1]  # the frames needed for this task
            keys = image_stack[needed_frames, :, :, :]
            flow = TemporalOpticalFlowSmoothing.worker((k, keys, device))

            flows.append(keys[0].cpu())
            for f in flow:
                flows.append(f[0].cpu())
            # flows.append(keys[1].cpu())

        return flows

    def handler(self, **kwargs):
        from copy import deepcopy
        # image stacks are actually (b/t,w,h,c) so we need to rearrange the dimensions
        k = kwargs["k"]
        image_stack = deepcopy(kwargs["image_stack"])
        device = kwargs.get("device", "cpu")

        image_stack = image_stack.permute(0, 2, 1, 3)

        # make sure image_stack is a torch tensor and has the right dimensions
        if not torch.is_tensor(image_stack):
            image_stack = torch.tensor(image_stack, dtype=torch.float32)

        if len(image_stack.shape) < 4:  # if it doesn't have a channel dimension
            image_stack = image_stack.unsqueeze(0)  # add a batch dimension

        # Move image stack to the specified device
        image_stack = image_stack.to(device)

        # Calculate optical flow
        flows = self.calculate_optical_flow(image_stack, device, k)
        stack = torch.stack(flows, dim=0)
        # Convert optical flows to images for visualization
        # flow_images = [flow_to_image(flow) for flow in flows]
        # change back to (b or t,w,h,c)

        stack = stack.permute(0, 2, 1, 3)
        return (stack,)


class MovingCircle(metaclass=ABCWidgetMetaclass):
    # VideoWidget class definition

    def calculate_start(self, x1, y1, width, height):
        # Convert to 0-1 range
        start_x = x1 * width
        start_y = y1 * height
        return (start_x, start_y)

    def calculate_end(self, x2, y2, width, height):
        # Convert to 0-1 range
        end_x = x2 * width
        end_y = y2 * height
        return (end_x, end_y)

    def generate_frames(self, num_frames, width, height):
        # Match tensor shape from example
        frames = torch.zeros(num_frames, width, height, 3)
        return frames

    def draw_circle_or_oval(self, kp, width, height, radius, aspect_ratio, color=(255, 255, 255)):
        import torch.nn.functional as F
        # Initialize black background
        img = torch.zeros((3, height, width), dtype=torch.uint8)

        # Draw circle as keypoints
        keypoints = torch.tensor([[kp[0], kp[1]]]).unsqueeze(0)
        img = draw_keypoints(img, keypoints, colors=color, radius=radius)

        # convert to (w,h,c)
        # img = img.permute(1, 2, 0).unsqueeze(0)
        img = img.unsqueeze(0)

        # Calculate new size
        if aspect_ratio < 1:
            new_size = (height, round(width / aspect_ratio))
        else:
            new_size = (round(height * aspect_ratio), width)

        # Stretch the image
        img_stretched = F.interpolate(img, size=new_size, mode='bilinear', align_corners=False)

        # Shrink it back
        img_shrunk = F.interpolate(img_stretched, size=(height, width), mode='bilinear', align_corners=False)

        # Convert back to float values

        img_shrunk = img_shrunk.float() / 255

        img_shrunk = img_shrunk.permute(0, 2, 3, 1)
        return img_shrunk

    def draw_circle(self, kp, width, height, radius, color=(255, 255, 255)):
        # Initialize black background
        img = torch.zeros((3, height, width), dtype=torch.uint8)
        # Draw circle as keypoints
        keypoints = torch.tensor([[kp[0], kp[1]]]).unsqueeze(0)
        img = draw_keypoints(img, keypoints, colors=color, radius=radius)
        # convert to (w,h,c)
        img = img.permute(1, 2, 0).unsqueeze(0)
        # convert back to float values
        img = img.float() / 255

        return img

    # Metadata for documentation
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ETK/video"
    FUNCTION = "handler"

    # Match parent class method
    @classmethod
    def INPUT_TYPES(cls):
        # Validate types, ranges
        return {"required":
            {
                "x1": ("FLOAT", {"min": 0, "max": 1, "step": 0.1}),
                "y1": ("FLOAT", {"min": 0, "max": 1, "step": 0.1}),
                "x2": ("FLOAT", {"min": 0, "max": 1, "step": 0.1}),
                "y2": ("FLOAT", {"min": 0, "max": 1, "step": 0.1}),
                "num_frames": ("INT", {"min": 1}),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
                "radius": ("INT", {"min": 1}),
                "color": ("STRING", {"default": "255 255 255"}),
                "aspect_ratio": ("FLOAT", {"min": 0.1, "max": 10})

            }}

    def handler(self, x1, y1, x2, y2, num_frames, width, height, radius, color="255 255 255", aspect_ratio=1):
        import torch
        # Avoid errors on bad values
        assert 0 <= x1 <= 1
        assert 0 <= y1 <= 1

        # Split calculations
        start = torch.tensor(self.calculate_start(x1, y1, width, height))

        # Split calculations
        end = torch.tensor(self.calculate_end(x2, y2, width, height))

        # color is supposed to be a tuple
        color = tuple(map(lambda x: int(x), color.split()))
        # Initialize for compositing

        frames = []
        for i in range(num_frames):
            # Leverage PyTorch
            cur_pos = start + (end - start) * i / num_frames
            circle = self.draw_circle_or_oval(cur_pos, width, height, radius, aspect_ratio, color)
            # Use broadcasting to overlay
            frames.append(circle)

        # cat
        frames = torch.cat(frames, dim=0)

        return (frames,)

    if __name__ == "__main__":
        import doctest

        doctest.testmod()


class FindFrameFolders(metaclass=ABCWidgetMetaclass):
    """Finds all folders directly off a given root path containing sequential image files"""

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "root_path": ("STRING", {"default": "path to root of project"})
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("image_sequence_folders",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        root_path = kwargs["root_path"]
        image_sequence_folders = []

        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            if os.path.isdir(folder_path):
                image_files = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
                if len(image_files) >= 2:
                    image_sequence_folders.append(folder_path)

        return (image_sequence_folders,)


class CombineAudioAndVideoFiles(metaclass=ABCWidgetMetaclass):
    """Combines lists of audio files and video files using imageio-ffmpeg"""

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "audio_files": ("LIST", {"default": []}),
                "video_files": ("LIST", {"default": []}),
                "fps": ("FLOAT", {"default": 30, "min": 1, "max": 120}),
            },
            "optional": {
                "move_to_folder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("combined_video_paths",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        from os.path import splitext, join
        import shutil
        audio_files = kwargs["audio_files"]
        video_files = kwargs["video_files"]
        move_to_folder = kwargs.get("move_to_folder", "")

        combined_video_paths = []
        video_audio_pairs = zip(video_files, audio_files)

        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError("FFmpeg could not be found.")

        for video_file, audio_file in video_audio_pairs:
            video_out_file_fp = splitext(video_file)[0] + "_v_and_a.mp4"
            input_args = ['-i', video_file, '-i', audio_file]
            output_args = ['-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-y', video_out_file_fp]
            command = [ffmpeg_exe] + input_args + output_args
            os.system(' '.join(command))

            if move_to_folder:
                destination_path = join(move_to_folder, os.path.basename(video_out_file_fp))
                shutil.move(video_out_file_fp, destination_path)
                combined_video_paths.append(destination_path)
            else:
                combined_video_paths.append(video_out_file_fp)

        return (combined_video_paths,)


class CombineFoldersWithFramesToVideoFiles(metaclass=ABCWidgetMetaclass):
    """Combines lists of folders containing frames of individual video files using imageio and ffmpeg to create video files"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folders_with_frames": ("LIST", {"default": []}),
                "fps/length": ("FLOAT", {"default": 30, "min": 1, "max": 10000000000}),
                "bitrate": ("STRING", {"default": "20000k"}),
                "codec": (
                    ["libx264", "libx265", "mpeg4", "vp9", "huffyuv", "flv1"],
                    {"default": "libx264"}
                ),
                "file_extension": (
                    [".mp4", ".mov", ".avi", ".flv", ".mkv", ".webm"],
                    {"default": ".mp4"}
                ),
                "fps or length": (["fps", "length"], {"default": "fps"}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING",)
    RETURN_NAMES = ("combined_video_paths",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        import imageio
        import os

        folders_with_frames = kwargs["folders_with_frames"]
        fps_length_value = kwargs["fps/length"]
        bitrate = kwargs["bitrate"]
        codec = kwargs["codec"]
        file_extension = kwargs["file_extension"]
        fps_or_length = kwargs["fps or length"]

        combined_video_paths = []
        messages = []

        for folder_path in folders_with_frames:
            try:
                out_file = os.path.join(os.path.dirname(folder_path),
                                        f"{os.path.basename(folder_path)}{file_extension}")
                frame_files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                if not frame_files:
                    messages.append(f"No frame files found in folder: {folder_path}")
                    continue

                if fps_or_length == "fps":
                    fps = fps_length_value
                else:
                    total_frames = len(frame_files)
                    fps = total_frames / fps_length_value if fps_length_value != 0 else 30

                writer = imageio.get_writer(out_file, fps=fps, codec=codec, bitrate=bitrate)

                for frame_file in frame_files:
                    frame_path = os.path.join(folder_path, frame_file)
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)

                writer.close()
                combined_video_paths.append(out_file)
            except Exception as e:
                messages.append(f"Error processing folder {folder_path}: {str(e)}")

        return (combined_video_paths, "\n".join(messages),)


class ImageStackToVideoFile(metaclass=ABCWidgetMetaclass):
    """Really just saves the image stack to a video file"""

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "video_out": ("STRING", {"multiline": False, "default": "video01.mp4"}),
                "fps or length": (["fps", "length"], {"default": "fps"}),
                "fps/length": ("FLOAT", {"default": 30, "min": 1, "max": 10000000000}),
                "bitrate": ("STRING", {"default": "20000k"}),
                "codec": (
                    ["libx264", "libx265", "mpeg4", "libvpx-vp9", "huffyuv", "flv1"],
                    {"default": "libx264"}
                ),
            },
            "optional":
                {
                    "audio_file": ("STRING", {"default": "NONE"}),
                },
        }

    RETURN_TYPES = ("FUNC", "STRING",)
    RETURN_NAMES = ("FUNC", "video_out_path",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        import os
        import numpy as np
        import uuid
        import shutil
        import imageio
        import audioread
        import cv2

        # video_dir = kwargs["video_dir"]

        audio_file = kwargs.get("audio_file")
        if audio_file == "NONE":
            audio_file = None

        image_stack = kwargs["image_stack"]

        fps_or_length = kwargs["fps or length"]
        fps_length_value = kwargs["fps/length"]
        bitrate = kwargs["bitrate"]
        codec = kwargs["codec"]
        video_out = kwargs["video_out"]

        if fps_or_length == "fps":
            fps_length_value = np.array([fps_length_value], dtype=np.float16)[0]

        video_out_path = video_out

        if os.path.exists(video_out_path):
            random_folder = os.path.join(os.path.dirname(video_out_path), str(uuid.uuid4()))
            os.makedirs(random_folder)
            shutil.move(video_out_path, random_folder)

        if audio_file is not None:
            audio_file = audio_file
            if not os.path.exists(audio_file):
                # raise Exception(f"Audio file {audio_file} does not exist")
                audio_length = 0
            else:
                with audioread.audio_open(audio_file) as audio_info:
                    audio_length = audio_info.duration

        frames_length = image_stack.shape[0]
        if fps_or_length == "fps":
            video_length = frames_length / fps_length_value
        else:
            video_length = fps_length_value

        if fps_or_length == "fps":
            video_fps = fps_length_value
        else:
            video_fps = frames_length / fps_length_value

        folder_out = "temp_frames_folder"

        # Now, read the frames from disk as numpy data
        import tempfile
        # use tempfile to create a temporary folder
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_out_path = temp_dir

            # Convert the image stack to frames on disk
            frame_folder_handler = ImageStackToVideoFramesFolder()
            frame_folder_handler.handler(image_stack=image_stack, folder_out=frames_out_path)

            frames = []
            for fl in sorted(os.listdir(frames_out_path)):
                # if the fl is an actual image file
                if not fl.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                # create the numpy array
                frame = cv2.imread(os.path.join(frames_out_path, fl))
                # convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # append to the list of frames
                frames.append(frame)

        writer = imageio.get_writer(video_out_path, fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        if audio_file is not None:
            if audio_length > video_length:
                while video_length < audio_length:
                    image_stack = np.concatenate((image_stack, image_stack[-1:]), axis=0)
                    video_length += 1 / video_fps
                    frames_length += 1

            args = [
                "-i", video_out_path,
                "-i", audio_file,
                "-c:v", codec,
                "-b:v", str(bitrate),
                "-c:a", "aac",
                "-strict", "experimental",
                video_out_path
            ]
            run_ffmpeg_command(args)

        return (self, video_out_path,)


class CreateConsistentVideo(metaclass=ABCWidgetMetaclass):
    """Creates a single video from a folder of videos with a consistent rate of change"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {"default": ""}),
                "rate_of_change_target": ("FLOAT", {"default": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "STRING",)
    RETURN_NAMES = ("joined_video_path", "individual_video_paths", "error_messages")
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        import asyncio
        import os

        video_folder = kwargs["video_folder"]
        rate_of_change_target = kwargs["rate_of_change_target"]

        individual_video_paths = []
        error_messages = []

        async def process_video(file_path):
            try:
                # Calculate visual differences
                difference = calculate_visual_difference(file_path)

                # Adjust frame rate based on the rate of change target
                adjusted_video_path = adjust_frame_rate(
                    video_path=file_path,
                    target_rate_of_change=rate_of_change_target,
                    visual_difference=difference

                )

                return adjusted_video_path

            except Exception as e:
                error_messages.append(str(e))
                return None

        # Gather video files

        video_files = [f for f in os.listdir(video_folder) if
                       f.lower().endswith(('.mp4', '.mov', '.avi', '.flv', '.mkv', '.webm'))]
        # now get their full path
        video_files = [os.path.join(video_folder, f) for f in video_files]

        # Use Asyncio to process videos in parallel
        # There is no current loop so we need to create a NEW one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processed_videos = loop.run_until_complete(asyncio.gather(
            *[process_video(file_path) for file_path in video_files]
        ))

        # Combine processed videos into the final video
        joined_video_path = combine_videos(processed_videos, video_folder, video_folder)

        return (joined_video_path, individual_video_paths, "\n".join(error_messages),)


def calculate_visual_difference(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open the video file")

    # Initialize variables to hold previous frame
    prev_frame_tensor = None
    total_difference = 0

    # Iterate through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to a PyTorch tensor and normalize
        frame_tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255

        # If previous frame exists, calculate the difference using SSIM
        if prev_frame_tensor is not None:
            difference = get_img_diff_torch(prev_frame_tensor, frame_tensor)
            total_difference += difference

        # Update previous frame
        prev_frame_tensor = frame_tensor

    # Close the video file
    cap.release()

    return total_difference


def get_img_diff_torch(image1: torch.Tensor, image2: torch.Tensor, K1=0.01, K2=0.03, L=255):
    # Convert images to grayscale using weighted average of color channels
    gray1 = 0.2989 * image1[0] + 0.5870 * image1[1] + 0.1140 * image1[2]
    gray2 = 0.2989 * image2[0] + 0.5870 * image2[1] + 0.1140 * image2[2]

    # Calculate mean and variance of input images
    mu1 = torch.mean(gray1)
    mu2 = torch.mean(gray2)
    var1 = torch.var(gray1)
    var2 = torch.var(gray2)
    covar = np.cov(gray1.cpu().numpy().ravel(), gray2.cpu().numpy().ravel())[0][1]

    # Set constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator

    # Return difference score
    return 1 - ssim


def adjust_frame_rate(video_path, target_rate_of_change, visual_difference):
    filename, _ = os.path.splitext(os.path.basename(video_path))
    base_dir = os.path.dirname(video_path)
    output_filename = f"{filename}_adjusted.mp4"
    output_filename = os.path.join(base_dir, output_filename)

    #  use the full path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open the video file")

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if visual_difference == 0:
        desired_fps = float(original_fps)
    else:
        desired_fps = max(float(original_fps * target_rate_of_change / visual_difference), 1.0)

    desired_fps = int(desired_fps * 8) / 8

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_filename, fourcc, desired_fps, frame_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    return output_filename


def combine_videos(video_paths, base_input_directory, output_path):
    import subprocess
    import tempfile

    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp_file:
        for video_path in video_paths:
            temp_file.write(f"file '{video_path}'\n")

    # TODO: Check cross-platform compatibility (linux)
    cmd = f"ffmpeg -f concat -safe 0 -i \"{temp_file.name}\" \"{output_path}combined{os.path.splitext(video_paths[0])[1]}\""

    try:
        subprocess.run(cmd, shell=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        return str(e)
    finally:
        # Clean up the temporary file
        temp_file.close()
        os.remove(temp_file.name)





class WhisperTranscription(metaclass=ABCWidgetMetaclass):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file_path": ("STRING", {"default": "fullpath"})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe_movie"
    CATEGORY = "transcription"

    def transcribe_movie(self, file_path):
        import os
        import time
        import numpy as np
        import whisper as wh
        import moviepy as mp
        try:
            model = wh.load_model("medium.en")
            filename = os.path.basename(file_path)
            audio_file_path = f"audio_{filename}.wav"

            video = mp.VideoFileClip(file_path)
            video.audio.write_audiofile(audio_file_path)

            res = wh.transcribe(model=model, audio=audio_file_path)
            segs = res["segments"]
            out = []
            for seg in segs:
                out.append(f"{seg['start']:3.3f} - {seg['end']:3.3f}: {seg['text']}")
            ret = '\n'.join(out)

            os.remove(audio_file_path)

            return (ret,)

        except Exception as e:
            print(e)
            return ("Error",)