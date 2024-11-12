# video_module.py
from .common import ETK_AVF_base, duplicate_file, run_ffmpeg_command


@ETK_AVF_base
class VideoFolderFFMPEGVideoInterpolate:
    """
    Use run_ffmpeg to interpolate the frames in the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
                "last_n_frames": ("INT", {"default": 2}),
                "target_n_frames": ("INT", {"default": 3}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    RETURN_NAMES = ("video_folder_def",)
    FUNCTION = "video_folder_ffmpeg_video_interpolate"

    def video_folder_ffmpeg_video_interpolate(self, **kwargs):
        import os
        import shutil

        video_folder_def = kwargs["video_folder_def"]
        last_n_frames = kwargs["last_n_frames"]
        target_n_frames = kwargs["target_n_frames"]

        # Get the current frame count
        current_frame_count = video_folder_def.get_frame_count()

        # Calculate the target frame rate for interpolation
        target_fps = video_folder_def["fps"] * (target_n_frames / last_n_frames)
        source_fps = video_folder_def["fps"]

        # Get the file names for the last n frames
        last_frame_names = [video_folder_def.get_last_file_name(offset=i) for i in range(last_n_frames)]
        last_frame_names = list(set(last_frame_names))
        last_frame_names.sort()

        # Create a temporary folder off of the video frame folder's base path
        temp_folder = os.path.join(video_folder_def["video_folder"], "temp")
        os.makedirs(temp_folder, exist_ok=True)

        # Move the input frames into the new folder
        for frame_name in last_frame_names:
            # Do this in a way that overwrites the files and MOVES them
            shutil.move(frame_name, os.path.join(temp_folder, os.path.basename(frame_name)))

        # New input frame names list
        last_frame_names = [os.path.join(temp_folder, os.path.basename(frame_name)) for frame_name in last_frame_names]
        last_frame_names.sort()

        # Create a file with the list of input frame paths in the temporary folder
        input_frame_list_file = os.path.join(temp_folder, "input_frames.txt")
        input_frame_list_file = os.path.abspath(input_frame_list_file)
        with open(input_frame_list_file, 'w') as file:
            # Need to do this in a manner that puts the first frame at the top of the file.
            file.write('\n'.join([f"file '{frame_name}'" for frame_name in last_frame_names]))

        # Get the output frame names for the interpolated frames
        output_frame_names = [video_folder_def.get_next_file_name(i) for i in range(target_n_frames)]

        temp_output_frame_pattern = os.path.join(video_folder_def["video_folder"], "temp",
                                                 "tmp_frame_%010d." + video_folder_def["frame format"].lower())
        temp_output_frame_pattern = os.path.abspath(temp_output_frame_pattern)

        # Create the FFmpeg command for motion interpolation using the frame list file
        ffmpeg_interpolate_cmd = [
            "-f", "concat", "-safe", "0",
            "-r", str(source_fps),
            "-i", input_frame_list_file,
            "-vsync", "1",
            "-filter_complex",
            f"[0:v]setpts=N/TB,minterpolate=fps={target_fps}:mi_mode=blend,setpts=PTS-STARTPTS[v]",  # Reset timestamps
            "-map", "[v]", "-q:v", "2",
            "-r", str(target_fps),
            "-frames:v", str(target_n_frames),
            "-threads", "0",
            "-y", temp_output_frame_pattern
        ]

        run_ffmpeg_command(ffmpeg_interpolate_cmd)

        # Move the interpolated frames to the video folder with the correct names
        for i, frame_name in enumerate(output_frame_names):
            temp_frame_name = os.path.join(temp_folder,
                                           f"tmp_frame_{i + 1:010d}.{video_folder_def['frame format'].lower()}")
            try:
                shutil.move(temp_frame_name, frame_name)
            except FileNotFoundError:
                pass

        # Remove the temporary folder and frame list file
        os.remove(input_frame_list_file)
        # Remove this without generating an error even if it is not empty
        try:
            os.rmdir(temp_folder)
        except OSError:
            shutil.rmtree(temp_folder)

        return (video_folder_def,)


@ETK_AVF_base
class VideoDefinitionProvider:
    """
    Given a VIDEOFOLDER node, and inputs for height, width, and fps, creates a VIDEOFOLDERDEF variable
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder": ("VIDEOFOLDER", {"default": None}),
                "height": ("INT", {"default": 1080}),
                "width": ("INT", {"default": 1920}),
                "fps": ("FLOAT", {"default": 30}),
                "frame format": (["JPG", "PNG"], {"default": "JPG"}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    FUNCTION = "video_definition_provider"

    def video_definition_provider(self, **kwargs):
        from addict import Dict as adict
        import os

        ret = adict()
        ret["video_folder"] = kwargs["video_folder"]
        ret["height"] = kwargs["height"]
        ret["width"] = kwargs["width"]
        ret["fps"] = kwargs["fps"]
        ret["frame format"] = kwargs.get("frame format") or kwargs.get("frame_format")
        # A function that gets the frame count from the disk
        ret.get_frame_count = lambda: len([f for f in os.listdir(ret["video_folder"]) if "frame_" in f])
        # A function that gets the next file name and full path, don't forget to add 10 zeros to the frame count
        ret.get_next_file_name = lambda x=0: os.path.join(
            ret['video_folder'],
            f"frame_{str(ret.get_frame_count() + x).zfill(10)}.{ret['frame format'].lower()}"
        )
        ret.get_last_file_name = lambda offset=0: os.path.join(
            ret['video_folder'],
            f"frame_{str(max(ret.get_frame_count() - offset - 1, 0)).zfill(10)}.{ret['frame format'].lower()}"
        )
        ret.get_current_file_name = lambda: os.path.join(
            ret['video_folder'],
            f"frame_{str(ret.get_frame_count() - 1).zfill(10)}.{ret['frame format'].lower()}"
        )

        def reserve_next_frame():
            fn = ret.get_next_file_name()
            open(fn, "w").close()
            return fn

        ret.reserve_next_frame = reserve_next_frame

        return (ret,)


@ETK_AVF_base
class VideoFolderProvider:
    """
    Given a folder path, creates the folder or empties it entirely returns the folder path as
    a string but as the node type "VIDEOFOLDER"
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "clear_folder": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "activator:": ("*", {"default": None})
            }
        }

        return ret

    RETURN_TYPES = ("VIDEOFOLDER",)
    FUNCTION = "video_folder_provider"

    def video_folder_provider(self, **kwargs):
        import shutil
        import os

        # Check if the folder_path is none
        if kwargs["folder_path"] is None:
            raise ValueError("folder_path must be specified")

        # Set folder exists flag by using the os.path.exists function
        folder_exists = os.path.exists(kwargs["folder_path"])

        if not folder_exists:
            os.makedirs(kwargs["folder_path"])
            folder_exists = True

        # Set files in folder flag by using the os.listdir function
        files_in_folder = os.listdir(kwargs["folder_path"])

        # Set accessible flag by using the os.access function
        accessible = os.access(kwargs["folder_path"], os.W_OK)
        if not accessible:
            raise ValueError("Folder for video files is not accessible")

        if kwargs["clear_folder"]:
            # If the folder exists, delete all files in the folder
            if folder_exists:
                for file in files_in_folder:
                    # Use shutil to remove the file
                    fl = os.path.join(kwargs["folder_path"], file)
                    if os.path.isfile(fl):
                        os.remove(fl)
                    # Use shutil to remove the entire path
                    check_this_path = file
                    if len(check_this_path) > 3:
                        shutil.rmtree(check_this_path, ignore_errors=True)
                    else:
                        raise ValueError("Folder path is too short")

        return (kwargs["folder_path"],)


@ETK_AVF_base
class AddFramesToVideoFolder:
    """
    Given a VIDEO_FOLDER_DEF node, and a list of frames, adds the frames to the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
                "IMAGE": ("IMAGE", {"default": None}),
                "Rescale method": (
                    ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"],
                    {"default": "LANCZOS"}),
                "Release vram?": ("BOOLEAN", {"default": True}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    FUNCTION = "add_frames_to_video_folder"

    @staticmethod
    def save_or_copy_frame(current_frame, previous_frame, frame_fp, previous_frame_fp):
        import shutil
        from PIL import Image
        import torch

        if previous_frame is None or not torch.equal(current_frame, previous_frame):
            # If the current frame is different from the previous frame, save it to disk
            current_frame = current_frame
            # expected shape is H, W, C uint8 0-255
            # image.fromarray expects uint8 0-255 and shape H, W, C

            needs_dtype_conversion = current_frame.dtype != torch.uint8

            if len(current_frame.shape) == 3 and current_frame.dtype == torch.uint8:
                needs_dtype_conversion = False
            elif len(current_frame.shape) == 2:
                # assume grayscale H,W
                current_frame = current_frame.unsqueeze(2).repeat(1, 1, 3)
            elif len(current_frame.shape) == 4 and current_frame.shape[0] == 1:
                # assume B, H, W, C
                current_frame = current_frame.squeeze(0)
            elif len(current_frame.shape) == 3 and current_frame.shape[-1] == 4:
                # assume H, W, C with alpha
                current_frame = current_frame[..., :3]
            else:  # try to move forward
                pass
            # assume H, W, C

            if needs_dtype_conversion:
                current_frame = current_frame.mul(255).byte()

            ndarr = current_frame.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(frame_fp)

            previous_frame = current_frame
        else:
            # If the current frame is the same as the previous frame, copy the previously saved frame file
            shutil.copy2(previous_frame_fp, frame_fp)
        return previous_frame

    @staticmethod
    def process_frames(frames, rescale_method, video_folder_def):
        import gc
        import torch
        num_frames = frames.shape[0]
        previous_frame = None
        frame_fp = None
        frames_are_all_same = torch.all(frames[0] == frames)
        if frames_are_all_same and frames.shape[0] > 1:
            # Get a list of file names using reserve_next_frame
            frame_f_names = [video_folder_def.reserve_next_frame() for _ in range(num_frames)]
            # Save the first frame
            AddFramesToVideoFolder.save_or_copy_frame(frames[0], previous_frame, frame_f_names[0], None)
            # Copy the first frame to the rest of the frames using os by calling the cli
            duplicate_file(frame_f_names[0], frame_f_names[1:])

        else:
            for i in range(num_frames):
                next_fn = video_folder_def.get_next_file_name()

                previous_frame_fp = frame_fp
                frame_fp = next_fn

                current_frame = frames[i]

                previous_frame = AddFramesToVideoFolder.save_or_copy_frame(current_frame, previous_frame, frame_fp,
                                                                           previous_frame_fp)

        del frames
        gc.collect()
        return None

    def add_frames_to_video_folder(self, **kwargs):
        from PIL import Image
        import torch
        import gc
        from torchvision.transforms.functional import resize
        import psutil
        from custom_nodes.EternalKernelLiteGraphNodes.modules.image_utils import lanczos_resize

        initial_memory = psutil.virtual_memory().available
        print(f"Initial free memory: {initial_memory / (1024 * 1024):.2f} MB")

        # Get the video folder path
        video_folder = kwargs["video_folder_def"]["video_folder"]

        # Get the frame format
        frame_format = kwargs["video_folder_def"]["frame format"]

        # Get the height
        height = kwargs["video_folder_def"]["height"]

        # Get the width
        width = kwargs["video_folder_def"]["width"]

        # Get the fps
        fps = kwargs["video_folder_def"]["fps"]

        # Get the frames
        frames = kwargs["IMAGE"]

        # Get the rescale method
        rescale_method = kwargs.get("Rescale method", None)
        if not rescale_method:
            rescale_method = kwargs.get("Rescale_method", None)

        free_vram = kwargs.get("Release vram?", None)
        if not free_vram:
            free_vram = kwargs.get("Release_vram_", True)

        orig_frames = frames
        # num_files = len(glob(os.path.join(video_folder, f"frame_*.{frame_format.lower()}")))

        # Rescale if needed to the correct size
        if height != frames.shape[1] or width != frames.shape[2]:
            # First since transforms.Resize requires a tensor of shape B,C, H, W
            # We need to permute the tensor to B, C, H, W
            frames = frames.permute(0, 3, 1, 2)
            frames = frames.to("cuda")

            # Get the right value for interpolation
            interpolation_methods = {
                "NEAREST": Image.Resampling.NEAREST,
                "BOX": Image.Resampling.BOX,
                "BILINEAR": Image.Resampling.BILINEAR,
                "HAMMING": Image.Resampling.HAMMING,
                "BICUBIC": Image.Resampling.BICUBIC,
                "LANCZOS": Image.Resampling.LANCZOS
            }

            rescale_type = interpolation_methods[rescale_method]
            # rescale_method = transforms.Resize((height, width), interpolation=rescale_type) # this doesnt work so well lets use a different function

            if rescale_type != Image.Resampling.LANCZOS:
                rescale_method_fn = lambda x: resize(x, (height, width), interpolation=rescale_type)
                frames = rescale_method_fn(frames)
            else:
                rescale_method_fn = lambda x: lanczos_resize(x, (width, height))
                frames = rescale_method_fn(frames)

            # Now we need to permute back to B, H, W, C
            frames = frames.permute(0, 2, 3, 1)
            self.process_frames(frames, rescale_method_fn, kwargs["video_folder_def"])
        else:
            self.process_frames(frames, rescale_method, kwargs["video_folder_def"])

        gc.collect()

        if free_vram:
            import gc
            if isinstance(frames, torch.Tensor):
                frames.to("cpu").detach().numpy()
                del frames

            if isinstance(orig_frames, torch.Tensor):
                orig_frames = orig_frames.to("cpu").detach().numpy()
                del orig_frames

            torch.cuda.empty_cache()
            gc.collect()

        initial_memory = psutil.virtual_memory().available
        print(f"After adding frames free memory: {initial_memory / (1024 * 1024):.2f} MB")
        # Return the VIDEO_FOLDER_DEF
        return (kwargs["video_folder_def"],)


@ETK_AVF_base
class HoldFrameForSecsInVideoFolder:
    """
    Given a VIDEO_FOLDER_DEF node, and a list of frames, first uses RepeatFrames to repeat the frames
    for the correct amount and then adds the frames to the video folder using AddFramesToVideoFolder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
                "IMAGE": ("IMAGE", {"default": None}),
                "seconds": ("FLOAT", {"default": 1.0}),
                "Rescale method": (
                    ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"],
                    {"default": "LANCZOS"}),
                "Release vram?": ("BOOLEAN", {"default": True}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    FUNCTION = "hold_frame_for_secs_in_video_folder"

    def saver(self, x, kwargs):
        # This just saves in an async way
        rescale_method = kwargs.get("Rescale method", None)
        if rescale_method is None:
            rescale_method = kwargs.get("Rescale_method", None)

        rel_vram = kwargs.get("Release vram?", None)
        if rel_vram is None:
            rel_vram = kwargs.get("Release_vram_", True)

        AddFramesToVideoFolder().add_frames_to_video_folder(
            video_folder_def=kwargs["video_folder_def"],
            IMAGE=x,
            Rescale_method=rescale_method,
            Release_vram_=rel_vram
        )

    def hold_frame_for_secs_in_video_folder(self, **kwargs):
        video_folder_ref = kwargs["video_folder_def"]

        video_folder = kwargs["video_folder_def"]["video_folder"]

        # Get the frame format
        frame_format = kwargs["video_folder_def"]["frame format"]

        # Get the height
        height = kwargs["video_folder_def"]["height"]

        # Get the width
        width = kwargs["video_folder_def"]["width"]

        # Get the fps
        fps = kwargs["video_folder_def"]["fps"]

        # Get the frames
        frames = kwargs["IMAGE"]

        # Get the number of frames
        num_frames = frames.shape[0]

        # Get the rescale method
        rescale_method = kwargs.get("Rescale method", None)
        if rescale_method is None:
            rescale_method = kwargs.get("Rescale_method", None)

        # Get the number of seconds
        seconds = kwargs["seconds"]

        # Get the number of frames to hold
        num_frames_to_hold = int(fps * seconds)

        num_frames_in_this_stack = frames.shape[0]

        # Get the number of repeats needed
        num_repeats_needed = num_frames_to_hold // num_frames_in_this_stack

        # Get the number of frames left over
        num_frames_left_over = num_frames_to_hold % num_frames_in_this_stack

        # Write the first set of frames to disk
        self.saver(frames, kwargs)

        # Get the name of the last frame
        source_last_frame_name = video_folder_ref.get_last_file_name()
        # From the last frame find the rest of the frames names/paths
        source_last_frame_num = int(source_last_frame_name.split("_")[-1].split(".")[0])
        source_first_frame_num = source_last_frame_num - num_frames_in_this_stack + 1

        template = source_last_frame_name.replace(str(source_last_frame_num).zfill(10), "{}")

        range_of_source_frame_numbers = list(range(source_first_frame_num, source_last_frame_num + 1))

        source_frame_names = [template.format(str(x).zfill(10)) for x in range_of_source_frame_numbers]

        target_done_frame_num = source_last_frame_num + num_frames_to_hold - num_frames_in_this_stack
        num_frames_remaining = num_frames_to_hold - num_frames_in_this_stack
        target_start_frame_num = source_last_frame_num + 1

        target_frame_range = list(range(target_start_frame_num, target_done_frame_num + 1))

        target_frame_names = [template.format(str(x).zfill(10)) for x in target_frame_range]

        set_todo = {fn: [] for fn in source_frame_names}
        # We can now repeat the frames
        for i, target_frame_name in enumerate(target_frame_names):
            # Keep track of what stack frame we are on
            stack_frame_idx = i % num_frames_in_this_stack
            stack_frame_name_to_use = source_frame_names[stack_frame_idx]
            set_todo[stack_frame_name_to_use].append(target_frame_name)

        # Now we can repeat the frames
        for frame_name in set_todo:
            src_frame = frame_name
            target_frames = set_todo[src_frame]
            duplicate_file(src_frame, target_frames)

        return (kwargs["video_folder_def"],)


@ETK_AVF_base
class Listed_HoldFramesForSecsInVideoFolder:
    """This just calls HoldFrameForSecsInVideoFolder multiple times for a list of frames and a list of durations"""

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
                "IMAGE": ("LIST", {"default": None}),
                "seconds": ("LIST", {"default": None}),
                "Rescale method": (
                    ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"],
                    {"default": "LANCZOS"}),
                "Release vram?": ("BOOLEAN", {"default": True}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    FUNCTION = "listed_hold_frames_for_secs_in_video_folder"

    def listed_hold_frames_for_secs_in_video_folder(self, **kwargs):
        # The plan is to call HoldFrameForSecsInVideoFolder for each frame and duration
        # and then return the video_folder_def

        if "IMAGE" not in kwargs or "seconds" not in kwargs:
            return (kwargs["video_folder_def"],)

        for frame, duration in zip(kwargs["IMAGE"], kwargs["seconds"]):
            HoldFrameForSecsInVideoFolder().hold_frame_for_secs_in_video_folder(
                video_folder_def=kwargs["video_folder_def"],
                IMAGE=frame,
                seconds=duration,
                Rescale_method=kwargs["Rescale method"],
                Release_vram_=kwargs["Release vram?"])

        return (kwargs["video_folder_def"],)


@ETK_AVF_base
class VideoFolderDefInfo:
    """
    Returns everything in the VIDEO_FOLDER_DEF, and also returns information about the status of the folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF", "STRING", "INT", "INT", "FLOAT", "STRING", "INT", "LIST",)
    RETURN_NAMES = ("VFD", "video_folder", "height", "width", "fps", "frame format", "num_files", "list of files")

    OUTPUT_NODE = True
    FUNCTION = "video_folder_def_info"

    def video_folder_def_info(self, **kwargs):
        import os

        video_folder = kwargs["video_folder_def"]["video_folder"]
        height = kwargs["video_folder_def"]["height"]
        width = kwargs["video_folder_def"]["width"]
        fps = kwargs["video_folder_def"]["fps"]
        frame_format = kwargs["video_folder_def"]["frame format"]
        files = os.listdir(video_folder)
        num_files = len(files)

        return (kwargs["video_folder_def"], video_folder, height, width, fps, frame_format, num_files, files,)


@ETK_AVF_base
class FadeToInVideoFolder:
    """
    Given a VIDEO_FOLDER_DEF node, and a stack of frames, fades the frames in and adds them to the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {
                "video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
                "IMAGE": ("IMAGE", {"default": None}),
                "seconds": ("FLOAT", {"default": 1.0}),
                "transition ms": ("FLOAT", {"default": 0.1}),
                "Rescale method": (
                    ["NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"],
                    {"default": "LANCZOS"}),
                "Release vram?": ("BOOLEAN", {"default": True}),
            },
        }
        return ret

    RETURN_TYPES = ("VIDEO_FOLDER_DEF",)
    FUNCTION = "fade_to_in_video_folder"

    def fade_to_in_video_folder(self, **kwargs):
        from PIL import Image
        import torch

        # Use HoldFrameForSecsInVideoFolder to hold the frame after creating and AddFramesToVideoFolder the transition

        transition_length_secs = int(kwargs["transition ms"] * kwargs["video_folder_def"]["fps"]) / 1000
        hold_frame_length_secs = int(kwargs["seconds"] * kwargs["video_folder_def"]["fps"]) - transition_length_secs

        transition_frame_count = int(kwargs["video_folder_def"]["fps"] * transition_length_secs)

        base_frame_fn = kwargs["video_folder_def"].get_last_file_name()
        # Load it into a pytorch tensor (B, C, H, W)
        base_frame = Image.open(base_frame_fn)
        base_frame = torch.tensor(base_frame).permute(2, 0, 1).unsqueeze(0).to("cuda")

        transition_stack = torch.zeros(
            (transition_frame_count, base_frame.shape[1], base_frame.shape[2], base_frame.shape[3]),
            dtype=torch.float16, device="cuda"
        )

        # Fade from base_frame to the first frame in the stack over the transition length
        new_frame = kwargs["IMAGE"][0].to("cuda")
        for i in range(transition_frame_count):
            alpha = i / transition_frame_count
            interpolated_frame = (1 - alpha) * base_frame + alpha * new_frame
            interpolated_frame = interpolated_frame.to("cpu")
            transition_stack[i] = interpolated_frame

        # Add to the video folder
        AddFramesToVideoFolder().add_frames_to_video_folder(
            video_folder_def=kwargs["video_folder_def"],
            IMAGE=transition_stack,
            Rescale_method=kwargs["Rescale method"],
            Release_vram_=kwargs["Release vram?"]
        )

        # Hold the frame
        HoldFrameForSecsInVideoFolder().hold_frame_for_secs_in_video_folder(
            video_folder_def=kwargs["video_folder_def"],
            IMAGE=kwargs["IMAGE"],
            seconds=hold_frame_length_secs,
            Rescale_method=kwargs["Rescale method"],
            Release_vram_=kwargs["Release vram?"]
        )

        return (kwargs["video_folder_def"],)
