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

    RETURN_TYPES = ("FUNC", "STRING",)
    RETURN_NAMES = ("FUNC", "video_out_path",)
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
        try:
            self.CODE = inspect.getsource(ImageStackToVideoFile.handler)
        except OSError:
            self.CODE = "Unable to get source code"

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
        return (self, video_out_path + ".mp4",)


class TemporalSpatialSmoothing(metaclass=WidgetMetaclass):
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


import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
from torchvision.transforms.functional import resize


class TemporalOpticalFlowSmoothing(metaclass=WidgetMetaclass):
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


# TW: Inherit for code reuse, consistency
import torch
from torchvision.utils import draw_keypoints


class MovingCircle(metaclass=WidgetMetaclass):
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
        #img = img.permute(1, 2, 0).unsqueeze(0)
        img=img.unsqueeze(0)

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
        color = tuple(map(lambda x:int(x), color.split()))
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
