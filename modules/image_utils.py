import torch
import torch.nn as nn
import cv2
import time
from PIL import Image
import numpy as np
import scipy.ndimage


class MaskExpansionModel(nn.Module):
    """
    A PyTorch model that expands or erodes a mask using the provided function.
    """

    def forward(self, mask, expand, tapered_corners):
        assert mask.dim() == 4, "Input mask must have 3 dimensions (batch_size, height, width)."
        bs = mask.size(0)
        c = 0 if tapered_corners else 1
        kernel = torch.tensor([[c, 1, c],
                               [1, 1, 1],
                               [c, 1, c]], dtype=mask.dtype, device=mask.device)
        kernel = kernel.view(1, 1, 3, 3)

        if expand < 0:
            kernel = -kernel
            expand = -expand
            mask = 1 - mask

        mask = mask.unsqueeze(1)
        # print(mask.shape)
        for _ in range(expand):
            for i in range(bs):
                # print(i)
                mask[i] = torch.nn.functional.conv2d(mask[i], kernel, padding=1)
                mask[i] = torch.clamp(mask[i], 0, 1)  # Clip the mask values to the range [0, 1]

        mask = mask.squeeze(1)
        if expand < 0:
            mask = 1 - mask

        return mask


class LanczosInterpolationLayer(nn.Module):
    """
    A PyTorch module that applies Lanczos interpolation to resize a batch of images.

    Args:
        target_size (tuple): The target size for resizing the images, in the format (width, height).
    """

    def __init__(self, target_size):
        super(LanczosInterpolationLayer, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        """
        Performs the forward pass of the Lanczos interpolation layer.

        Args:
            x (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A batch of resized images with shape (batch_size, channels, target_height, target_width).
        """
        device = torch.device("cpu")
        interpolated_images = []
        for image in x:
            # Convert the PyTorch tensor to a NumPy array
            image_np = image.permute(1, 2, 0).to("cpu", torch.float32).numpy()

            # Resize the image using OpenCV's resize function with Lanczos interpolation
            interpolated_image = cv2.resize(image_np, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            mn = interpolated_image.min()
            mx = interpolated_image.max()

            color_range = mx - mn

            under_bottom_target_range = -10/255
            over_top_target_range = 1/255

            if color_range > 1:
                # here we are going to compress the over the top values
                over_top = interpolated_image > 1
                not_over_top = interpolated_image < 1
                over_top_range = mx - 1
                # subtract the normal top value
                interpolated_image[over_top] = interpolated_image[over_top] - 1
                # scale the over the top values to the target range
                interpolated_image[over_top] *= (over_top_target_range / over_top_range)
                # add the target range to the over the top values
                interpolated_image[over_top] += (1-over_top_target_range)
                # scale the non-over-the top to make room
                interpolated_image[not_over_top] *= (1 - over_top_target_range)

                # here we are going to compress the under the bottom values
                under_bottom = interpolated_image < 0
                not_under_bottom = interpolated_image > 0
                under_bottom_range = mn
                # subtract the normal bottom value
                interpolated_image[under_bottom] = interpolated_image[under_bottom] - 0
                # scale the under the bottom values to the target range
                interpolated_image[under_bottom] *= (under_bottom_target_range / under_bottom_range)
                # add the target range to the under the bottom values
                interpolated_image[under_bottom] += under_bottom_target_range
                # scale the non-under-the-bottom to make room
                interpolated_image[not_under_bottom] *= (1 - under_bottom_target_range)




            else:
                pass

            # Convert the resized image back to a PyTorch tensor
            # interpolated_image = torch.from_numpy(interpolated_image).to(device, dtype=torch.float16).permute(2, 0, 1)
            interpolated_image = torch.from_numpy(interpolated_image)

            # convert fix range to 0-255
            interpolated_image = interpolated_image * 255
            # convert to uint8
            interpolated_image = interpolated_image.to(device, dtype=torch.uint8).permute(2, 0, 1)

            interpolated_images.append(interpolated_image)
            del image
            del image_np
            del interpolated_image

        # transform this list of tensors into a single tensor of shape (batch_size, channels, height, width)

        # pre allocate a tensor of shape (batch_size, channels, target_height, target_width)
        y = torch.zeros((x.shape[0], interpolated_images[0].shape[0], self.target_size[1], self.target_size[0]),
                        device=device,
                        dtype=interpolated_images[0].dtype)
        y = torch.stack(interpolated_images, dim=0, out=y)
        del interpolated_images

        return y


class LanczosInterpolationModel(nn.Module):
    """
    A PyTorch model that applies Lanczos interpolation to resize a batch of images.

    Args:
        target_size (tuple): The target size for resizing the images, in the format (width, height).
    """

    def __init__(self, target_size):
        super(LanczosInterpolationModel, self).__init__()
        self.interpolation_layer = LanczosInterpolationLayer(target_size)

    def forward(self, x):
        """
        Performs the forward pass of the Lanczos interpolation model.

        Args:
            x (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A batch of resized images with shape (batch_size, channels, target_height, target_width).
        """
        return self.interpolation_layer(x)


def expand_mask(mask, expand, tapered_corners, device=None, dtype=None):
    """
    Convenience function for expanding or eroding a mask using the MaskExpansionModel.

    Args:
        mask (torch.Tensor): The input mask tensor.
        expand (int): The number of iterations to expand (positive) or erode (negative) the mask.
        tapered_corners (bool): Whether to use tapered corners when expanding or eroding the mask.
        device (torch.device, optional): The device to use for the model and input tensor.
                                         If not provided, it will use the default device (CPU).
        dtype (torch.dtype, optional): The data type to use for the input tensor.
                                       If not provided, it will use the default dtype (float32).

    Returns:
        torch.Tensor: The expanded or eroded mask tensor.
    """
    # Create an instance of the MaskExpansionModel
    model = MaskExpansionModel()

    # Move the model to the specified device (or default to CPU)
    device = device or torch.device("cpu")
    model = model.to(device)

    # Move the input tensor to the specified device and convert to the specified dtype
    dtype = dtype or torch.float32
    mask = mask.to(device, dtype=dtype)

    # Expand or erode the mask using the MaskExpansionModel
    with torch.no_grad():
        expanded_mask = model(mask, expand, tapered_corners)

    # Clear the GPU cache if using a GPU
    if device != torch.device("cpu"):
        torch.cuda.empty_cache()

    return expanded_mask


def lanczos_resize(image, target_size, device="cpu", dtype=torch.float32, output_type="tensor"):
    """
    Convenience function for resizing an image using the Lanczos interpolation model.

    Args:
        image (PIL.Image or torch.Tensor): The input image to be resized.
        target_size (tuple): The target size for resizing the image, in the format (width, height).
        device (torch.device, optional): The device to use for the model and input tensor.
                                         If not provided, it will use the default device (CPU).
        dtype (torch.dtype, optional): The data type to use for the input tensor.
                                       If not provided, it will use the default dtype (float32).
        output_type (str, optional): The type of the output image.
                                      If "tensor", the function will return a PyTorch tensor.
                                      If "pil", the function will return a PIL Image.
                                      If not provided, it will return a tensor by default.

    Returns:
        torch.Tensor or PIL.Image: The resized image.
    """
    import gc
    import os
    import sys
    if sys.platform == "win32":
        import psutil
        print(f"Free memory: {psutil.virtual_memory().available / 1024 / 1024} MB")
    else:
        # works on linux and mac
        print(f"Free memory: {os.popen('free -m').readlines()[1].split()[3]} MB")

    # Create an instance of the LanczosInterpolationModel
    model = LanczosInterpolationModel(target_size)

    # Move the model to the specified device (or default to CPU)
    device = device or torch.device("cpu")
    model = model.to(device)

    # Convert the input image to a PyTorch tensor
    if isinstance(image, list):
        if isinstance(image[0], Image.Image):
            try:
                images = np.array(image)
            except:
                images = np.array([np.array(img) for img in image])

            images = torch.from_numpy(images).to(device, dtype).permute(0, 3, 1, 2)
    elif isinstance(image, torch.Tensor):
        images = image.to(device, dtype)

    del image
    torch.cuda.empty_cache()

    # Resize the image using the Lanczos interpolation model
    with torch.no_grad():
        resized_images = model(images)

    # Convert the resized tensor back to a PIL Image or keep it as a tensor based on the output_type
    if output_type.lower() == "pil":
        resized_images = resized_images.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        resized_images = Image.fromarray(resized_images.astype(np.uint8))
    elif output_type.lower() == "tensor":
        pass  # keep the resized_image as a tensor
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Valid options are 'tensor' or 'pil'.")

    # Clear the GPU cache if using a GPU
    if device != torch.device("cpu"):
        torch.cuda.empty_cache()

    del images
    del model
    gc.collect()
    # print current free memory on cpu, in a way that is cross platform 1 way that works on all

    if sys.platform == "win32":
        import psutil
        print(f"Free memory: {psutil.virtual_memory().available / 1024 / 1024} MB")
    else:
        # works on linux and mac
        print(f"Free memory: {os.popen('free -m').readlines()[1].split()[3]} MB")

    return resized_images


def test_lanczos_interpolation_model():
    """
    Tests the Lanczos interpolation model by resizing a batch of random images and benchmarking the performance.
    """
    # Set the target sizes for resizing
    hd_size = (1920, 1080)
    sd_size = (1280, 720)

    # Generate random PIL images
    batch_size = 600
    hd_images = [Image.fromarray(np.random.randint(0, 256, size=(hd_size[1], hd_size[0], 3), dtype=np.uint8))
                 for _ in range(batch_size)]
    sd_images = [Image.fromarray(np.random.randint(0, 256, size=(sd_size[1], sd_size[0], 3), dtype=np.uint8))
                 for _ in range(batch_size)]

    # Perform the benchmarks
    num_iterations = 3

    # HD to SD benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        sd_output_images = lanczos_resize(hd_images, sd_size)
    end_time = time.time()
    hd_to_sd_elapsed_time = end_time - start_time
    hd_to_sd_avg_time_per_iteration = hd_to_sd_elapsed_time / num_iterations

    # SD to HD benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        hd_output_images = lanczos_resize(sd_images, hd_size)
    end_time = time.time()
    sd_to_hd_elapsed_time = end_time - start_time
    sd_to_hd_avg_time_per_iteration = sd_to_hd_elapsed_time / num_iterations

    # Check the sizes of the output images

    assert sd_output_images.shape[0] == batch_size

    assert hd_output_images.shape[0] == batch_size

    print(f"Lanczos interpolation test passed!")
    print(f"Benchmark results:")
    print(f"Number of iterations: {num_iterations}")
    print(f"HD to SD:")
    print(f"  Total elapsed time: {hd_to_sd_elapsed_time:.4f} seconds")
    print(f"  Average time per iteration: {hd_to_sd_avg_time_per_iteration:.4f} seconds")
    print(f"  Average time per HD image -> SD image: {hd_to_sd_avg_time_per_iteration / batch_size:.4f} seconds")
    print(f"SD to HD:")
    print(f"  Total elapsed time: {sd_to_hd_elapsed_time:.4f} seconds")
    print(f"  Average time per iteration: {sd_to_hd_avg_time_per_iteration:.4f} seconds")
    print(f"  Average time per SD image -> HD image: {sd_to_hd_avg_time_per_iteration / batch_size:.4f} seconds")


def test_mask_expansion_model():
    """
    Tests the MaskExpansionModel by expanding and eroding random masks and benchmarking the performance.
    """
    # Generate random masks
    batch_size = 1000
    mask_size = (512, 512)
    # masks are 0-1 tensors float
    masks = torch.rand((batch_size, mask_size[0], mask_size[1])).to("cuda")
    print(masks.min(), masks.max())

    # Perform the benchmarks
    num_iterations = 1
    expand_steps = 5
    erode_steps = -5

    # Mask expansion benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        print(masks.mean())
        expanded_masks = expand_mask(masks, expand_steps, tapered_corners=True, device="cuda")
        print(masks.mean())
    end_time = time.time()
    expansion_elapsed_time = end_time - start_time
    expansion_avg_time_per_iteration = expansion_elapsed_time / num_iterations

    # Mask erosion benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        print(masks.mean())
        eroded_masks = expand_mask(masks, erode_steps, tapered_corners=True, device="cuda")
        print(eroded_masks.mean())
    end_time = time.time()
    erosion_elapsed_time = end_time - start_time
    erosion_avg_time_per_iteration = erosion_elapsed_time / num_iterations

    # Check the sizes of the output masks
    expected_mask_size = (batch_size, mask_size[0], mask_size[1])
    assert expanded_masks.shape == expected_mask_size, f"Expected expanded mask size: {expected_mask_size}, but got: {expanded_masks.shape}"
    assert eroded_masks.shape == expected_mask_size, f"Expected eroded mask size: {expected_mask_size}, but got: {eroded_masks.shape}"

    print(f"Mask expansion/erosion test passed!")
    print(f"Benchmark results:")
    print(f"Number of iterations: {num_iterations}")
    print(f"Mask expansion:")
    print(f"  Total elapsed time: {expansion_elapsed_time:.4f} seconds")
    print(f"  Average time per iteration: {expansion_avg_time_per_iteration:.4f} seconds")
    print(f"  Average time per mask expansion: {expansion_avg_time_per_iteration / batch_size:.4f} seconds")
    print(f"Mask erosion:")
    print(f"  Total elapsed time: {erosion_elapsed_time:.4f} seconds")
    print(f"  Average time per iteration: {erosion_avg_time_per_iteration:.4f} seconds")
    print(f"  Average time per mask erosion: {erosion_avg_time_per_iteration / batch_size:.4f} seconds")


if __name__ == "image_utils":
    print("image_utils module loaded.")
    print("Testing Lanczos interpolation model...")
    test_lanczos_interpolation_model()
    print("Testing Mask expansion model...")
    # test_mask_expansion_model()
    exit(0)

if __name__ == "__main__":
    test_lanczos_interpolation_model()
    # test_mask_expansion_model()
