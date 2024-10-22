### info for code completion AI ###
"""
all of these classes are plugins for comfyui and follow the same pattern
all of the images are torch tensors and it is unknown and unimportant if they are on the cpu or gpu
all image inputs are (B,W,H,C)

avoid numpy and PIL as much as possible
"""
import shutil
import fractions
from abc import ABC, ABCMeta, abstractmethod
from os.path import splitext, join
import torchaudio
import os
import torch
from torchvision.utils import draw_keypoints
import numpy as np
import cv2
import imageio_ffmpeg as ffmpeg

from . import NODE_CLASS_MAPPINGS

# from . import NODE_DISPLAY_NAME_MAPPINGS

# NODE_CLASS_MAPPINGS = {}  # this is the dictionary that will be used to register the nodes

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
    if isinstance(args, str):
        args = args.split()
    cmd = [ffmpeg.get_ffmpeg_exe()] + args
    subprocess.run(cmd)


# start with an ABC to define the common widget interface

class ABCWidgetMetaclass(ABCMeta):
    """A metaclass that automatically registers classes."""

    def __init__(cls, name, bases, attrs):
        if (ABC in bases) or (ABCVideoWidget in bases) or ("ABC" in name):
            pass
        else:
            NODE_CLASS_MAPPINGS[str(name)] = cls
            cls.CATEGORY = "ETK/video"
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
                "func_only": ([False, True], {"default": False}),
            },

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
        import time
        start = time.time()
        video_in_path = os.path.join(self.input_dir, video_in)
        video_out_path = os.path.join(self.output_dir, folder_out)

        vidcap = cv2.VideoCapture(video_in_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_out_path, "frame%d.jpg" % count), image)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
        end = time.time()
        print("time elapsed: ", end - start)
        return (folder_out,)


class VideoToFramesFolderFFMPEG(ABCVideoFileToABCVideoFolder, metaclass=ABCWidgetMetaclass):
    """use ffmpeg to open the video and use it as a source for image stacks, use PNG or jpeg with quality 100"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "video_in": ("STRING", {"default": "full_path_to_folder"}),
                "folder_out": ("STRING", {"default": "full_path_to_folder"}),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
            },

        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder_out",)

    def __init__(self):
        super().__init__()
        self.old = self.INPUT_TYPES
        # self.INPUT_TYPES = lambda s: self.old().update({})

    def handler(self, video_in, folder_out, format):
        import time
        start = time.time()
        video_in_path = os.path.join(self.input_dir, video_in)
        video_out_path = os.path.join(self.output_dir, folder_out)

        if not os.path.exists(video_out_path):
            os.makedirs(video_out_path)

        # specifically this calls ffmpeg with the following arguments:
        # -i means input file
        # -qscale:v means quality scale video, default is 2, which is a good compromise between size and quality
        # 1 is the quality scale value, which is a value between 1 and 31, where 1 is the highest quality

        start = time.time()

        if format == "JPEG":
            subprocess.call(
                ["ffmpeg", "-i", video_in_path, "-qscale:v", "1", os.path.join(video_out_path, "frame%05d.jpg")])
            end = time.time()
            print("ffmpeg jpg process took {} seconds".format(end - start))

        elif format == "PNG":
            subprocess.call(["ffmpeg", "-i", video_in_path, os.path.join(video_out_path, "frame%05d.png")])
            end = time.time()
            print("ffmpeg png process took {} seconds".format(end - start))

        else:
            raise ValueError("format must be either PNG or JPEG")

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
    RETURN_TYPES = ("IMAGE", "LIST",)
    RETURN_NAMES = ("image_stack", "[img_fname]",)

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
        image_fname_list = []

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
            image_fname_list.append(os.path.basename(fn))
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
        return (tensor, image_fname_list,)


class SmoothStackTemporalByDistance2(ABCABCVideoFolderToImage, metaclass=ABCWidgetMetaclass):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "top_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 1.0}),
                "bottom_thresh_normed": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 1.0}),
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
                "method": (["SSIM", "MSE", "REGIONAL"], {"default": "SSIM"}),
            },

        }

    RETURN_TYPES = ("STRING", "LIST", "IMAGE",)

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
            # find the max possible value of the mean squared error

        elif method == "REGIONAL":
            """
            this method resamples the image stack of B,W,H,C to B,8,8,C
            then calculates the mean squared error of each region of the image
            by further slicing into [b,1,1,C] x 64 regional representations
            the greatest mean squared from each region is returned
            """
            # the input image stack is defined as "image_stack" and it is a torch tensor, B,H,W,C

            B, H, W, C = image_stack.shape
            levels = [3, 4, 5, 6, 7, 8, 16, 32, 64]
            max_mse_value_in_all_regions_and_frames_for_this_level = 0
            level_stats = {k: {"max_mse": None, "max_std_dev": None} for k in levels}
            for size in levels:
                # need to re order the channels for the interpolation
                resampled_stack = image_stack.permute(0, 3, 1, 2)

                # Resample the image stack to 8x8 pixels
                resampled_stack = torch.nn.functional.interpolate(resampled_stack, size=(size, size), mode='bilinear',
                                                                  align_corners=False)

                # but the channel order back to the original
                resampled_stack = resampled_stack.permute(0, 2, 3, 1)

                # Reshape the resampled stack to (B, 64, C)
                regional_stack = resampled_stack.view(B, size ** 2, C)

                # Calculate the mean squared difference between consecutive frames for each region
                mse_values = torch.mean((regional_stack[1:] - regional_stack[:-1]) ** 2, dim=2)

                # max mse for each frame
                max_mse_ea_frame_at_level = torch.max(mse_values, dim=1)[0]
                max_dev_ea_frame_at_level = torch.std(mse_values, dim=1)

                # Calculate the standard deviation of the mean squared error for each region along the time dimension
                max_std_dev_of_all_regions_over_time = torch.max(torch.std(mse_values, dim=0))

                # in terms of video
                # represents the maximum amount of change in the image stack, basically the maximum amount of motion
                max_mse_value_in_all_regions_and_frames_for_this_level = torch.max(mse_values.view(-1))

                level_stats[size]["max_r_mse_ea_frame"] = max_mse_ea_frame_at_level
                level_stats[size]["max_r_dev_ea_frame"] = max_dev_ea_frame_at_level
                level_stats[size]["max_mse"] = max_mse_value_in_all_regions_and_frames_for_this_level
                level_stats[size]["max_std_dev"] = max_std_dev_of_all_regions_over_time

            all_l_max_mse_ea_frame = [level_stats[k]["max_r_mse_ea_frame"].tolist() for k in levels]
            all_l_max_std_ea_frame = [level_stats[k]["max_r_dev_ea_frame"].tolist() for k in levels]

            max_l_r_mse_per_batch = []  # Initialize list to store the maximum mean squared error for each batch_num
            for batch_num in range(B - 1):
                max_value = float('-inf')  # Initialize max_value to negative infinity for each batch_num
                for level in levels:
                    mse_for_level_and_batch = level_stats[level]["max_r_mse_ea_frame"][batch_num]
                    # Update max_value if the current mse_for_level_and_batch is greater
                    if mse_for_level_and_batch > max_value:
                        max_value = mse_for_level_and_batch
                max_l_r_mse_per_batch.append(
                    max_value)  # Append the maximum value for the current batch_num to the list

            max_l_r_dev_per_batch = []  # Initialize list to store the maximum standard deviation for each batch_num
            for batch_num in range(B - 1):
                max_value = float('-inf')
                for level in levels:
                    dev_for_level_and_batch = level_stats[level]["max_r_dev_ea_frame"][batch_num]
                    if dev_for_level_and_batch > max_value:
                        max_value = dev_for_level_and_batch
                max_l_r_dev_per_batch.append(max_value)

            # all_f_max_mse_ea_frame = [level_stats[k]["max_mse_ea_frame"] for k in levels]

            max_mse_value_in_all_regions_and_frames_for_all_levels = torch.max(torch.tensor(all_l_max_mse_ea_frame))
            max_std_dev_value_in_all_regions_and_frames_for_all_levels = torch.max(torch.tensor(all_l_max_std_ea_frame))

            out_str = f"max difference between any region over time: {max_mse_value_in_all_regions_and_frames_for_this_level}\nmax of how much the amount of motion changes over time: {max_std_dev_of_all_regions_over_time}"
            out_list = [max_mse_value_in_all_regions_and_frames_for_all_levels,
                        max_std_dev_value_in_all_regions_and_frames_for_all_levels,
                        max_l_r_mse_per_batch,
                        max_l_r_dev_per_batch
                        ]

            # return the resampled image stack

            out_image = None

            return (out_str, out_list, out_image,)



        else:
            # not implimented
            raise NotImplementedError

        image_stack_diff_len = len(stack_mse_B)
        required_image_info_width = 512 + image_stack_diff_len

        mean = torch.mean(stack_mse_B)
        std = torch.std(stack_mse_B)
        median = torch.median(stack_mse_B)
        max_mse = torch.max(stack_mse_B)
        min_mse = torch.min(stack_mse_B)

        out_str = f"mean: {mean}\n std: {std}\n median: {median}\n max: {max_mse}\n min: {min_mse}\n len: {image_stack_diff_len}"

        # create the histogram
        hist_values, _ = np.histogram(stack_mse_B.numpy(), bins=min(100, image_stack_diff_len))
        hist_values = hist_values / hist_values.max()  # Normalize to [0, 1]

        # Create an image tensor with shape (1, 512, required_width, 3) filled with zeros
        out_hist_image = torch.zeros((1, 512, required_image_info_width, 3))

        # Determine the scaling factors for width and height for the histogram
        scale_width = 512 // len(hist_values)
        scale_height = out_hist_image.shape[1]

        # Fill the image tensor with the histogram
        for i, value in enumerate(hist_values):
            height = 512 - int(value * scale_height)
            start_x = int(i * scale_width)
            end_x = int((i + 1) * scale_width)
            out_hist_image[0, height:, start_x:end_x - 1, :] = 1  # Color as white

        # repeat each element of the stack_mse_B tensor by scale_width
        stack_mse_B = stack_mse_B.repeat_interleave(scale_width)

        # Create a plot for the difference for each frame pair
        plot_offset = 512
        for i, value in enumerate(stack_mse_B):
            height = 512 - int(value * (scale_height / .5))  # Normalize by the max value
            start_x = plot_offset + i
            end_x = plot_offset + i + 1
            out_hist_image[0, height:, start_x:end_x, :] = (1 * height) / 512  # Color as white

        return (out_str, out_hist_image,)


# @profile
class VideoFileToImageStack(ABCABCVideoFileToImage, metaclass=ABCWidgetMetaclass):
    """Use cv2 to open a video and save the videos individual frames"""
    RETURN_TYPES = ("IMAGE", "FUNC",)
    RETURN_NAMES = ("image_stack", "FUNC(**kwargs)",)

    def handler(self, video_in, idx_slice_start, idx_slice_stop, slice_idx_step, idx, func_only=False):
        """
        Return slices of time from the video, sequences of images, defined by the start, stop and step
        - torch tensors (T,H,W,C)
        """
        import sys
        import gc
        # print memory usage
        print(f"Memory usage: {sys.getsizeof(self)} bytes")

        if func_only:
            return (None, VideoFileToImageStack.handler,)

        image_list = []
        video_in_fp = video_in
        vidcap = cv2.VideoCapture(video_in_fp)
        max_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not vidcap.isOpened():
            print("Error: Unable to open video file")
            return

        start = (idx * slice_idx_step) + idx_slice_start
        stop = (idx * slice_idx_step) + idx_slice_stop
        slice_length = stop - start

        if (start + slice_length) > max_frames:
            raise ValueError(f"idx_slice_stop ({idx_slice_stop}) is greater than the max frames ({max_frames})")

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
            print(f"Memory usage: {sys.getsizeof(self)} bytes")

        vidcap.release()
        # Convert list of images to tensor (T,H,W,C)
        try:
            tensor = torch.stack([torch.tensor(img, dtype=torch.float32) for img in image_list])
        except RuntimeError as e:
            print("Error: Unable to convert image list to tensor")
            raise e

        tensor = tensor / 255.0
        del image_list
        del vidcap
        del image
        gc.collect()
        # print memory usage
        print(f"Memory usage: {sys.getsizeof(self)} bytes")
        return (tensor, VideoFileToImageStack.handler,)


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

    RETURN_TYPES = ("STRING",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
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
        return (folder_out,)


class SmoothStackTemporal(ABCABCVideoFolderToImage, metaclass=ABCWidgetMetaclass):
    """
    Smooth the temporal dimension of the image stack by calculating the distance between each frame
    and the next frame and then averaging the frames that are within a certain distance of each other.
    Supports adaptive mode with target deviation and baseline mode.
    Can also perform frame interpolation using RIFE VFI.
    """

    rife_kwargs = {
        "ckpt_name": "rife49.pth",
        "clear_cache_after_n_frames": 1000,
        "multiplier": 2,
        "fast_mode": False,
        "ensemble": False,
        "scale_factor": 1.0,
        "optional_interpolation_states": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_stack": ("IMAGE",),
                "top_threshold": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 1.0}),
                "bottom_threshold": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 1.0}),
                "max_iters": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 1}),
                "target_deviation": ("FLOAT", {"min": 0.0, "max": 100000.0, "step": 0.01, "default": 0.0}),
                "interp_frames": ("BOOLEAN", {"default": False}),
                "mode": (["adaptive", "baseline"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")

    def compute_stats(self, image_stack):
        stats_handler = GetImageStackStatistics().handler
        res = stats_handler(image_stack=image_stack, method="REGIONAL")
        stats_str, stats_list, _ = res
        region_diff = torch.tensor(stats_list[2])  # The list of each frame pairs maximum regional difference
        max_diff = stats_list[0]  # Overall maximum difference
        max_dev = stats_list[1]  # Overall regional deviation
        return stats_str, region_diff, max_diff, max_dev

    def compute_baseline_motion(self, region_diff):
        values, counts = torch.unique(region_diff, return_counts=True)
        baseline_motion = values[torch.argmax(counts)].item()
        return baseline_motion

    def normalize_diff(self, region_diff):
        region_diff_range = region_diff.max() - region_diff.min()
        if region_diff_range == 0:
            region_diff_range = 1  # Avoid division by zero
        region_diff_norm = (region_diff - region_diff.min()) / region_diff_range
        return region_diff_norm

    def process_top_outliers(self, image_stack, region_diff_norm, top_threshold, rife_vfi):
        import torch.nn.functional as F

        top_outliers = torch.where(region_diff_norm > top_threshold)[0]
        new_image_stack = []

        for idx in range(image_stack.shape[0]):
            new_image_stack.append(image_stack[idx])
            if idx in top_outliers and idx < image_stack.shape[0] - 1:  # Ensure we don't go out of bounds
                frame_pair = image_stack[idx:idx + 2]
                interpolated_frames = rife_vfi(frames=frame_pair, **self.rife_kwargs)[0]

                # Calculate the MSE for each interpolated frame with the first and last frames of the pair
                mse_first = [F.mse_loss(interpolated_frame, frame_pair[0]) for interpolated_frame in
                             interpolated_frames]
                mse_last = [F.mse_loss(interpolated_frame, frame_pair[1]) for interpolated_frame in interpolated_frames]

                # Find the interpolated frame with the smallest difference between mse_first and mse_last
                mse_diff = [abs(mse_f - mse_l) for mse_f, mse_l in zip(mse_first, mse_last)]
                middle_frame_index = torch.argmin(torch.tensor(mse_diff)).item()

                choice = interpolated_frames[middle_frame_index]
                new_image_stack.append(choice)

        new_image_stack = torch.stack(new_image_stack)
        return new_image_stack, len(top_outliers)

    def drop_bottom_outliers(self, image_stack, region_diff_norm, bottom_threshold):
        bottom_outliers = torch.where(region_diff_norm < bottom_threshold)[0]
        if len(bottom_outliers) > 0:
            mask = torch.ones(image_stack.size(0), dtype=torch.bool)
            mask[bottom_outliers + 1] = False
            image_stack = image_stack[mask]
        return image_stack, len(bottom_outliers)

    def log_info(self, iteration, top_outliers_count, bottom_outliers_count, mean_diff, std_diff, max_diff, max_dev,
                 stats_str):
        info = f"Iteration {iteration}:\n"
        info += f"Frames changed by top threshold: {top_outliers_count}\n"
        info += f"Frames dropped by bottom threshold: {bottom_outliers_count}\n"
        info += f"Mean of region_diff: {mean_diff}\n"
        info += f"Std of region_diff: {std_diff}\n"
        info += f"Overall max difference: {max_diff}\n"
        info += f"Overall max deviation: {max_dev}\n"
        info += f"Image stack statistics: {stats_str}\n\n"
        return info

    def handler(self, image_stack, top_threshold=0.9, bottom_threshold=0.10, max_iters=1, target_deviation=0.0,
                interp_frames=False, mode="adaptive"):
        import torch
        import nodes

        image_stack = image_stack.to(torch.float16)
        log = ""
        iteration = 0

        if interp_frames:
            rife_vfi = nodes.NODE_CLASS_MAPPINGS["RIFE VFI"]().vfi

        if mode == "baseline":
            # Compute initial statistics and baseline motion
            stats_str, region_diff, max_diff, max_dev = self.compute_stats(image_stack)
            baseline_motion = self.compute_baseline_motion(region_diff)
            target_deviation = baseline_motion
            log += f"Baseline motion (mode of max regional differences): {baseline_motion}\n"

        while True:
            # Step 1: Compute statistics
            stats_str, region_diff, max_diff, max_dev = self.compute_stats(image_stack)

            # Step 2: Normalize differences
            region_diff_norm = self.normalize_diff(region_diff)

            mean_diff = region_diff.mean().item()
            std_diff = region_diff.std().item()

            # Step 3: Check if the target deviation is met or if max iterations are reached
            if std_diff <= target_deviation or iteration >= max_iters:
                log += f"std_diff <= target_deviation or iteration >= max_iters"
                break

            # Step 4: Identify and process top outliers
            image_stack, top_outliers_count = self.process_top_outliers(image_stack, region_diff_norm, top_threshold,
                                                                        rife_vfi)

            # Step 5: Identify and drop bottom outliers
            image_stack, bottom_outliers_count = self.drop_bottom_outliers(image_stack, region_diff_norm,
                                                                           bottom_threshold)

            # Log information
            log += self.log_info(iteration, top_outliers_count, bottom_outliers_count, mean_diff, std_diff, max_diff,
                                 max_dev, stats_str)

            # Increment the iteration counter
            iteration += 1

            # Step 6: Check if the image stack is empty
            if image_stack.shape[0] == 0:
                return (None, "Image stack is empty after processing. No frames left.")

        return (image_stack.to(torch.float32), log)


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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_files": ("LIST", {"default": []}),
                "video_files": ("LIST", {"default": []}),
                "resample": (["None", "video", "audio", ],),
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

    @classmethod
    def has_audio_stream(cls, video_file):
        """Check if the video file has an audio stream."""
        import subprocess
        command = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_name', '-of',
                   'default=nw=1:nk=1', video_file]
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            return bool(output.strip())
        except subprocess.CalledProcessError:
            return False

    def get_duration(self, file):
        """Get the duration of a file in seconds."""
        import subprocess

        command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                   'default=nw=1:nk=1', file]
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            return float(output.strip())
        except subprocess.CalledProcessError as e:
            print (f"Could not determine duration of file: {file}")
            print (f"Command: {command}")
            raise e

    def handler(self, **kwargs):
        import subprocess

        audio_files = kwargs["audio_files"]
        video_files = kwargs["video_files"]
        move_to_folder = kwargs.get("move_to_folder", "")
        resample = kwargs.get("resample", "None")
        fps = kwargs.get("fps", 30.0)

        combined_video_paths = []
        video_audio_pairs = list(zip(video_files, audio_files))

        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError("FFmpeg could not be found.")

        for video_file, audio_file in video_audio_pairs:
            video_file = f'{video_file}'
            audio_file = f'{audio_file}'

            video_out_file_fp = f'{splitext(video_file)[0]}_v_and_a.mp4'
            input_args = ['-i', video_file, '-i', audio_file]

            if self.has_audio_stream(video_file):
                # The video has an audio stream, proceed with mixing
                filter_complex_arg = '[0:a][1:a]amix=inputs=2:duration=longest[a]'
                output_args = ['-c:v', 'copy', '-map', '0:v:0', '-map', '[a]', '-c:a', 'aac', '-strict', 'experimental',
                               '-y', video_out_file_fp]
            else:
                filter_complex_arg = ''
                output_args = ['-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-strict',
                               'experimental', '-y', video_out_file_fp]

            if resample == "video":
                video_duration = self.get_duration(video_file)
                audio_duration = self.get_duration(audio_file)

                if video_duration and audio_duration:
                    speed = audio_duration / video_duration
                    filter_complex_arg = f'[0:v]setpts={speed}*PTS[v];[v][1:a]concat=n=1:v=1:a=1[v][a]'
                    output_args = ['-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-c:a', 'aac', '-strict',
                                   'experimental', '-y', video_out_file_fp]
                else:
                    print(f"audio duration: {audio_duration}, video duration: {video_duration}")
                    raise RuntimeError("Could not determine duration of video or audio file.")
            elif resample == "None":
                output_args.extend(['-r', str(fps)])
            elif resample == "audio":
                # Not implemented: Finding silence in audio and removing it
                raise NotImplementedError(
                    "Resampling audio is not implemented. Consider finding silence in the audio and removing it.")

            try:
                # try to run it with the list of parts, using the list
                if filter_complex_arg:
                    commands_to_run = [ffmpeg_exe, *input_args, '-filter_complex', filter_complex_arg, *output_args]
                else:
                    commands_to_run = [ffmpeg_exe, *input_args, *output_args]

                commands_to_run = [str(c).replace("\"", "") for c in commands_to_run]
                subprocess.run(commands_to_run, check=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

            if move_to_folder:
                destination_path = join(move_to_folder, os.path.basename(video_out_file_fp))

                if video_out_file_fp != destination_path:
                    shutil.move(video_out_file_fp, destination_path)

                combined_video_paths.append(destination_path)
            else:
                combined_video_paths.append(video_out_file_fp)

            os.chdir("..")
            os.chdir("..")
            os.chdir("..")

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
                    ["libx264", "libx265", "mpeg4", "vp9", "huffyuv", "flv1", "h264_nvenc", "hevc_nvenc", "rawvideo"],
                    {"default": "libx264"}
                ),
                "file_extension": (
                    [".mp4", ".mov", ".avi", ".flv", ".mkv", ".webm"],
                    {"default": ".mp4"}
                ),
                "fps or length": (["fps", "length"], {"default": "fps"}),
            },
            "optional": {
                "[duration or fps]": ("LIST", {"default": []}),
                "q": ("INT", {"default": 20, "min": 0, "max": 31}),
                "pix_fmt": ("STRING", {"default": "yuv420p"}),
            },
        }

    RETURN_TYPES = ("LIST", "STRING",)
    RETURN_NAMES = ("combined_video_paths",)
    CATEGORY = "video"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        folders_with_frames = kwargs.get("folders_with_frames", [])
        fps_length_value = kwargs.get("fps/length", 30)
        file_extension = kwargs.get("file_extension", ".mkv")
        fps_or_length = kwargs.get("fps or length", "fps")
        duration_or_fps_list = kwargs.get("[duration or fps]", [])
        codec = kwargs.get("codec", "libx264")
        q = kwargs.get("q", 20)
        pix_fmt = kwargs.get("pix_fmt", "yuv420p")

        combined_video_paths = []
        messages = []

        ffmpeg_exe = 'ffmpeg'  # Ensure ffmpeg is in your PATH or provide the full path to the ffmpeg executable
        if not shutil.which(ffmpeg_exe):
            raise RuntimeError("FFmpeg could not be found.")

        for i, folder_path in enumerate(folders_with_frames):
            try:
                output_file = os.path.join(os.path.dirname(folder_path),
                                           f"{os.path.basename(folder_path)}tmp_{i}{file_extension}")
                if fps_or_length == 'fps':
                    fps = fps_length_value
                    if duration_or_fps_list:
                        fps = duration_or_fps_list[i]
                else:
                    frame_files = sorted(
                        [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    )
                    total_frames = len(frame_files)
                    fps = total_frames / fps_length_value if fps_length_value != 0 else 30
                    if duration_or_fps_list:
                        fps = total_frames / duration_or_fps_list[i]

                create_video_from_images(folder_path, output_file, fps, codec, q, pix_fmt)
                combined_video_paths.append(output_file)
            except Exception as e:
                import traceback

                # Capture full traceback information
                full_traceback = traceback.format_exc()
                messages.append(f"Error processing folder '{folder_path}': {str(e)}\nFull Traceback:\n{full_traceback}")

        return (combined_video_paths, "\n".join(messages),)


def create_file_list(folder_path):
    frame_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )
    list_file_path = os.path.join(folder_path, 'filelist.txt')
    with open(list_file_path, 'w') as f:
        for frame_file in frame_files:
            f.write(f"file '{os.path.join(folder_path, frame_file)}'\n")
    return list_file_path


def create_video_from_images(folder_path, output_file, fps, codec, q, pix_fmt):
    import os
    import subprocess
    import shutil
    import traceback

    # Assuming create_file_list is part of the larger framework, not redefining it.
    list_file_path = create_file_list(folder_path)

    # Ensure ffmpeg is available in PATH
    ffmpeg_exe = shutil.which('ffmpeg')
    if ffmpeg_exe is None:
        raise RuntimeError("FFmpeg is not found. Please ensure it is installed and available in the system's PATH.")

    # Normalize paths for cross-platform compatibility
    list_file_path = os.path.normpath(list_file_path)
    output_file = os.path.normpath(output_file)

    # FFmpeg command to create the video
    command = [
        ffmpeg_exe, '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path,
        '-vf', f"fps={fps}", '-c:v', codec, '-pix_fmt', pix_fmt, '-qp', str(q), output_file
    ]

    try:
        # Capture both stdout and stderr to diagnose errors
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Capture the full traceback and the error output from the ffmpeg process
        error_message = e.stderr.decode() if e.stderr else "No additional error information"
        full_traceback = traceback.format_exc()
        raise RuntimeError(f"ffmpeg failed with error: {error_message}\nFull Traceback:\n{full_traceback}")


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
        import torchvision

        use_torchvision = False

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

        if use_torchvision == False:
            writer = imageio.get_writer(video_out_path, fps=float(video_fps))

            # convert the image stack to uint8
            image_stack = (image_stack * 255).to(torch.uint8)

            for frame_data in image_stack:
                frame = np.array(frame_data)  # Convert torch tensor to numpy array if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)

            if audio_file is not None:
                difference = video_length - audio_length

                needed_extra_frames = int(difference * video_fps)

                if needed_extra_frames > 0:
                    for i in range(needed_extra_frames):
                        writer.append_data(frame)

                writer.close()

                args = [
                    "-i", video_out_path,
                    "-i", audio_file,
                    "-c:v", codec,
                    "-b:v", str(bitrate),
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-y",
                    video_out_path.replace(".mp4", "_with_audio.mp4")
                ]
                run_ffmpeg_command(args)
            else:
                writer.close()

        if use_torchvision:
            audio_tensor = None
            if audio_file is not None:
                audio_tensor, aud_sr = torchaudio.load(audio_file)
            fps_frac = fractions.Fraction(float(video_fps))
            torchvision.io.write_video(filename=video_out_path,
                                       video_array=image_stack * 255,
                                       fps=fps_frac,
                                       audio_array=audio_tensor,
                                       video_codec=codec,
                                       audio_codec="aac",
                                       audio_fps=aud_sr
                                       )

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


def transcribe_to_segs(input_filename):
    import whisper as wh
    import folder_paths as fp

    if not os.sep in input_filename:
        input_dir = fp.input_directory
        full_path = os.path.join(input_dir, input_filename)
    else:
        full_path = input_filename
    # check that the file exists

    if not os.path.exists(full_path):
        raise ValueError(f"File {full_path} does not exist.")
    # check that it is a file
    if not os.path.isfile(full_path):
        raise ValueError(f"{full_path} is not a file.")

    model = wh.load_model("medium.en")

    res = wh.transcribe(model=model, audio=full_path, word_timestamps=True)
    segs = res["segments"]

    return segs


class WhisperTranscription(metaclass=ABCWidgetMetaclass):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_filename": ("STRING", {"default": "fullpath"})}}

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lines_out", "words_out",)
    FUNCTION = "transcribe_movie"
    CATEGORY = "transcription"

    def transcribe_movie(self, input_filename):

        full_path = os.path.abspath(input_filename)

        try:
            segs = transcribe_to_segs(full_path)

            lines_out = []
            for seg in segs:
                lines_out.append(f"{seg['start']:3.3f} - {seg['end']:3.3f}: {seg['text']}")

            lines_out = '\n'.join(lines_out)

            words_out = []
            for seg in segs:
                words_out.append([])
                for word in seg['words']:
                    line = f"{word['start']:3.3f} - {word['end']:3.3f}: {word['word']}"
                    words_out[-1].append(line)

            words_out = '\n'.join(['\n'.join(x) for x in words_out])

            return (lines_out, words_out,)

        except Exception as e:
            print(e)
            return ("Error", str(e))


class WhisperTranscribeToFrameIDXandWords(metaclass=ABCWidgetMetaclass):
    """Just like WhisperTranscription, but returns a list of frame numbers and words instead of a string."""

    @classmethod
    def INPUT_TYPES(cls):
        # needs full path to video file
        # needs fps for conversion
        ret = {"required": {"input_filename": ("STRING", {"default": "fullpath"}),
                            "fps": ("INT", {"default": 30})}}
        return ret

    RETURN_TYPES = ("LIST", "LIST", "LIST",)
    RETURN_NAMES = ("frame_idx", "words_out", "combined",)
    FUNCTION = "transcribe_movie"
    CATEGORY = "transcription"

    def transcribe_movie(self, input_filename, fps):
        import folder_paths as fp
        input_dir = fp.input_directory
        full_path = os.path.join(input_dir, input_filename)
        try:
            segs = transcribe_to_segs(full_path)
            frame_idx = []
            words_out = []
            combined = []
            for seg in segs:
                for word in seg['words']:
                    start_frame = int(word['start'] * fps)
                    end_frame = int(word['end'] * fps)
                    word_text = word['word']

                    frame_idx.append((start_frame, end_frame))
                    words_out.append(word_text)
                    combined.append((start_frame, end_frame, word_text))

            # now we fill in the gaps so the entire video is covered by some word
            # we do this by finding if there is a gap then changing the frame stamps of the last word and the next word
            # to cover the gap
            for i in range(len(frame_idx) - 1):
                if frame_idx[i][1] < frame_idx[i + 1][0]:
                    # there is a gap
                    gap_len = frame_idx[i + 1][0] - frame_idx[i][1]
                    to_previous_word = gap_len // 2
                    to_next_word = gap_len - to_previous_word

                    frame_idx[i] = (frame_idx[i][0], frame_idx[i][1] + to_previous_word)
                    frame_idx[i + 1] = (frame_idx[i + 1][0] - to_next_word, frame_idx[i + 1][1])

                    combined[i] = (combined[i][0], combined[i][1] + to_previous_word, combined[i][2])
                    combined[i + 1] = (combined[i + 1][0] - to_next_word, combined[i + 1][1], combined[i + 1][2])

                    # leave words_out alone

            return (frame_idx, words_out, combined,)
        except Exception as e:
            print(e)
            return ("Error",)


# @profile
def get_text_frames(word_idx, words_data, x: callable, y: callable, size: (), text_kwargs: dict, style="default"):
    from .image import TextRender as text_render_class
    text_render = text_render_class().render_text

    start = words_data[word_idx][0]
    end = words_data[word_idx][1]
    word_text = words_data[word_idx][2]
    frame_length = end - start

    # if x and y are not callables
    if not callable(x):
        x = lambda i: x
    if not callable(y):
        y = lambda i: y

    out_frames = []
    for i in range(start, end):
        it_dict = dict()
        it_dict["text"] = word_text
        it_dict["x"] = x(i / float(frame_length))
        it_dict["y"] = y(i / float(frame_length))
        it_dict["width"] = size[0]
        it_dict["height"] = size[1]

        it_dict.update(text_kwargs)
        # it_dict["size"] = 128
        # it_dict["color"] = "#ffffff"
        # it_dict["func_only"] = False
        # it_dict["stroke fill"] = "#050505"
        # it_dict["stroke width"] = 5

        text_img = text_render(**it_dict)
        out_frames.append(text_img[0])
    return out_frames


class AddTranscriptionToVideo(metaclass=ABCWidgetMetaclass):
    """
    Add a transcription to a video.  Requires a video and transcription data.
    transcription data must be [(start_frame, end_frame, word), ...]
    """

    @classmethod
    def INPUT_TYPES(cls):
        # needs full path to video file
        # needs fps for conversion
        ret = {"required": {"video_file": ("STRING", {"default": "fullpath"}),
                            "words": ("LIST", {"default": None}),
                            }
               }
        return ret

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("1 fps preview", "Full Video")
    FUNCTION = "add_transcription_to_video"
    CATEGORY = "transcription"

    # @profile
    def add_transcription_to_video(self, **kwargs):
        import torch
        import gc

        video_file = kwargs["video_file"]
        words = kwargs["words"]

        def create_video_frames_folder(video_file):
            fnc = VideoToFramesFolderFFMPEG().handler
            call_dict = dict()
            call_dict["video_in"] = video_file
            call_dict["folder_out"] = os.path.join(video_file, os.sep, "frames_tmp")
            call_dict["format"] = "PNG"

            tmp_video_folder = fnc(**call_dict)[0]
            del fnc
            del call_dict
            gc.collect()
            return tmp_video_folder

        def get_video_frames(start, end):
            fnc = VideoFramesFolderToImageStack().handler
            call_dict = dict()
            call_dict["use_subfolders"] = False
            call_dict["by_date_or_name"] = "NAME"
            call_dict["folder_in"] = os.path.join(video_file, os.sep, "frames_tmp")
            call_dict["idx_slice_start"] = 0
            call_dict["idx_slice_stop"] = end - start
            call_dict["slice_idx_step"] = 1
            call_dict["idx"] = start
            call_dict["use_float16"] = False

            ret = fnc(**call_dict)[0]

            return ret

        def overlay_text_on_video(video_frames, text_frames):
            # Create a mask for the text frames (where text is black)
            text_mask = (text_frames == 0) * 1  # Assuming black is represented as 0 in text_frames

            # Use the ~ operator to invert the boolean mask
            # inverted_mask = text_mask

            # Overlay text_frames on video_frames using the inverted_mask
            overlaid_frames = video_frames * text_mask + text_frames

            return overlaid_frames

        frames_out = []
        num_frames = 0
        frames_folder = create_video_frames_folder(video_file)
        for w_idx in range(len(words)):
            word_start = words[w_idx][0]
            word_end = words[w_idx][1]
            word = words[w_idx][2].strip()
            print(f"word_start: {word_start}, word_end: {word_end}, word: {word}")

            words_video_frames = get_video_frames(word_start, word_end)
            i_width = words_video_frames.shape[2]
            i_height = words_video_frames.shape[1]
            text_frames = get_text_frames(word_idx=w_idx,
                                          words_data=words,
                                          x=lambda i: 0,
                                          y=lambda i: 0,
                                          size=(i_width, i_height),
                                          text_kwargs={"size": 128, "color": "#ffffff", "func_only": False,
                                                       "stroke fill": "#050505", "stroke width": 5,
                                                       "font_name": "c:/Windows/Fonts/arial.ttf"})

            text_frames = torch.cat(text_frames)

            # Overlay text on video frames
            video_frames_with_text = overlay_text_on_video(words_video_frames, text_frames)
            frames_out.append(video_frames_with_text)
            num_frames += video_frames_with_text.shape[0]
            gc.collect()

        # free memory
        del words_video_frames
        del text_frames
        del video_frames_with_text

        num_stks = len(frames_out)
        # write to disk
        for i, stk in enumerate(frames_out):
            torch.save(stk, f"tmp_str_{i}.pt")
        del frames_out
        gc.collect()
        torch.cuda.empty_cache()

        # pre allocate tensor
        y = torch.zeros((num_frames, i_height, i_width, 3), dtype=torch.float32, device="cpu")
        # load each save and put into tensor
        new_start = 0
        for i in range(num_stks):
            stk = torch.load(f"tmp_str_{i}.pt")
            y[new_start:new_start + stk.shape[0]] = stk
            new_start = new_start + stk.shape[0]
            del stk
        gc.collect()
        torch.cuda.empty_cache()

        preview = y[::30].clone()

        return (preview, y,)


class TransformImageStack(metaclass=ABCWidgetMetaclass):
    """
    Transform a stack of images
    given x change (in pixels)
    given y change (in pixels)


    """

    # TODO: support sheer,rotate,scale
    # TODO: support transparancy
    # TODO: support fill color specification

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"ImageStack": ("IMAGE", {"default": None}),
                            "pan x": ("INT", {"default": 0, "min": -2048}),
                            "pan y": ("INT", {"default": 0, "min": -2048}),
                            },
               }
        return ret

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_image_stack"

    def transform_image_stack(self, **kwargs):
        import torchvision.transforms.functional as TF

        x_image = kwargs["ImageStack"]
        x_image = x_image.detach().clone()
        x = kwargs["pan x"]
        y = kwargs["pan y"]

        time_index, H, W, C = x_image.shape  # Assuming x_image has shape (time_index, H, W, C)
        out_list = []

        stop_x = x
        stop_y = y
        dx_step = stop_x / time_index - 1
        dy_step = stop_y / time_index - 1

        for t in range(time_index):
            tx = dx_step * t
            ty = dy_step * t

            current_frame = x_image[t].detach().clone()
            # Change from (H, W, C) to (C, H, W)
            current_frame = current_frame.permute(2, 0, 1)

            transformed_frame = TF.affine(current_frame, angle=0, translate=(tx, ty), scale=1,
                                          shear=[0, 0]).detach().clone()

            # Convert back to (H, W, C)
            transformed_frame = transformed_frame.permute(1, 2, 0)

            # Store in the list
            x_image[t] = transformed_frame

        return (x_image.detach().clone(),)


class PanByStep(metaclass=ABCWidgetMetaclass):
    """
    Transform a stack of images
    given x change (in pixels)
    given y change (in pixels)
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"ImageStack": ("IMAGE", {"default": None}),
                            "pan x": ("INT", {"default": 0, "min": -2048}),
                            "pan y": ("INT", {"default": 0, "min": -2048}),
                            "wrap": ("BOOLEAN", {"default": False}),
                            },
               }
        return ret

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pan_by_step"

    def pan_by_step(self, **kwargs):
        import torchvision.transforms.functional as TF

        x_image = kwargs["ImageStack"]
        x_image = x_image.detach().clone()
        x = kwargs["pan x"]
        y = kwargs["pan y"]

        wrap = kwargs["wrap"]

        time_index, H, W, C = x_image.shape  # Assuming x_image has shape (time_index, H, W, C)
        out_list = []

        stop_x = x
        stop_y = y
        dx_step = 0 if x == 0 else stop_x / time_index - 1
        dy_step = stop_y / time_index - 1

        for t in range(time_index):
            tx = dx_step * t
            ty = dy_step * t

            current_frame = x_image[t].detach().clone()
            # Change from (H, W, C) to (C, H, W)
            current_frame = current_frame.permute(2, 0, 1)

            # transformed_frame = TF.affine(current_frame, angle=0, translate=(tx, ty), scale=1,
            #                              shear=[0, 0]).detach().clone()

            # If wrap is True, apply the wrapping operation
            if wrap:
                transformed_frame = current_frame.roll(
                    (int(tx) % W, int(ty) % H), dims=(2, 1)
                )
            else:
                transformed_frame = TF.affine(current_frame, angle=0, translate=(tx, ty), scale=1,
                                              shear=[0, 0]).detach().clone()

            # Convert back to (H, W, C)
            transformed_frame = transformed_frame.permute(1, 2, 0)

            # Store in the list
            x_image[t] = transformed_frame

        return (x_image.detach().clone(),)


class RepeatFrames(metaclass=ABCWidgetMetaclass):
    """
    does nothing more than repeat the frames a number of times
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"IMAGE": ("IMAGE", {"default": None}),
                            "repeat": ("INT", {"default": 1}),
                            },
               }
        return ret

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("repeated images",)
    FUNCTION = "repeat_frames"

    def repeat_frames(self, **kwargs):
        frames = kwargs["IMAGE"]
        repeat = kwargs["repeat"]
        return (frames.repeat(repeat, 1, 1, 1),)


class ListedSVDConditioning(metaclass=ABCWidgetMetaclass):
    """
    repeats SVD_img2vid_Conditioning for each set of params
    """

    @classmethod
    def INPUT_TYPES(cls):
        from comfy_extras.nodes_video_model import SVD_img2vid_Conditioning
        base = SVD_img2vid_Conditioning.INPUT_TYPES()
        base["required"]["init_image"] = ("LIST", {"default": None})
        # base["required"]["clip_vision"] = ("LIST", {"default": None})
        base["required"]["video_frames"] = ("LIST", {"default": None})

        return base

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "listed_SVD_conditioning"

    def listed_SVD_conditioning(self, **kwargs):
        from comfy_extras.nodes_video_model import SVD_img2vid_Conditioning

        # just like Listed_HoldFramesForSecsInVideoFolder, we will call SVD_img2vid_Conditioning for each set of params

        positives = []
        negatives = []
        latents = []

        init_images = kwargs["init_image"]
        clip_vision = kwargs["clip_vision"]
        # vae
        vae = kwargs["vae"]
        # width and height
        width = kwargs["width"]
        height = kwargs["height"]
        video_frames = kwargs["video_frames"]
        motion_bucket_id = kwargs["motion_bucket_id"]
        fps = kwargs["fps"]
        augmentation_level = kwargs["augmentation_level"]

        lens = [len(init_images), len(video_frames)]

        # check if one length is divisible by the other
        if lens[0] % lens[1] != 0 and lens[1] % lens[0] != 0:
            raise ValueError("The length of init_image and video_frames must be divisible by each other")

        # repeat the shorter
        if lens[0] < lens[1]:
            init_images = init_images * (lens[1] // lens[0])
        else:
            video_frames = video_frames * (lens[0] // lens[1])

        # at this point the lengths should be the same

        for init_image, video_frame in zip(init_images, video_frames):
            pos, neg, latent = SVD_img2vid_Conditioning().encode(
                clip_vision=clip_vision,
                init_image=init_image,
                video_frames=video_frame,
                vae=vae,
                width=width,
                height=height,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                augmentation_level=augmentation_level
            )

            positives.append(pos)
            negatives.append(neg)
            latents.append(latent)

        return (positives, negatives, latents,)


class AddMetadataNode(metaclass=ABCWidgetMetaclass):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"default": "input.mp4"}),
                "output_file": ("STRING", {"default": "output.mp4"}),
                "metadata": ("STRING", {"default": "title:My Video"}),
            }
        }

    RETURN_TYPES = ("STRING",)  # This node returns a string message indicating success or failure
    RETURN_NAMES = ("status_message",)  # Providing a name for the return type
    FUNCTION = "add_metadata"
    CATEGORY = "video processing"

    def add_metadata(self, input_file, output_file, metadata):
        from ffmpeg import FFmpeg
        import shutil

        import torch  # Importing torch as it seems to be a convention in your framework
        # Parse the metadata from string to dictionary
        # strip input and output file
        input_file = input_file.strip()
        output_file = output_file.strip()

        # so actually if the input_file and output_file are the same
        # it will by default have an error
        # co if this is the case copy then input file to a tempfile then use that as the input file
        import tempfile

        if input_file == output_file:
            # copy the input file to a temp file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_name = temp_file.name
                shutil.copy(input_file, temp_file_name)
                input_file = temp_file_name

        metadata_dict = {}
        try:
            for line in metadata.splitlines():
                key, value = line.split(':', 1)  # Split on the first colon
                mdkey = "metadata"
                # metadata_dict.append(f'metadata:{key.strip()}={value.strip()}')
                metadata_dict[mdkey] = f'{key.strip()}={value.strip()}'

            # Run ffmpeg to add metadata
            ffmpeg = FFmpeg().option("y")
            try:
                # make sure to tell it to overwrite outputfile

                ffmpeg.input(input_file).output(output_file, **metadata_dict).execute()

                return (f"status_message : Metadata added successfully to {output_file}.",)
            except Exception as e:
                return (f"status_message : Error adding metadata to {output_file}.{str(e)}",)

            return (f"status_message : Metadata added successfully to {output_file}.",)
        except Exception as e:
            return (f"status_message : Error adding metadata to {output_file}.",)

    @staticmethod
    def describe():
        return "This node adds metadata to video files using ffmpeg. Metadata should be provided as 'key: value' pairs separated by newlines."
