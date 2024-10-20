from . import NODE_CLASS_MAPPINGS
from . import NODE_DISPLAY_NAME_MAPPINGS


def ETK_AVF_base(cls):
    cls.CATEGORY = "ETK/AudioVideoFolders"
    # Add spaces to the camel case class name
    pretty_name = cls.__name__
    for i in range(1, len(pretty_name)):
        if pretty_name[i].isupper():
            pretty_name = pretty_name[:i] + " " + pretty_name[i:]
    cls.DISPLAY_NAME = pretty_name
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name

    return cls


@ETK_AVF_base
class VideoFolderFFMPEGVideoInterpolate():
    """
    use run_ffmpeg to interpolate the frames in the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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

        # Get the file names for the last n frames
        last_frame_names = [video_folder_def.get_last_file_name(offset=i) for i in range(last_n_frames)]
        last_frame_names = list(set(last_frame_names))
        last_frame_names.sort()

        # Create a temporary folder off of the video frame folder's base path
        temp_folder = os.path.join(video_folder_def["video_folder"], "temp")
        os.makedirs(temp_folder, exist_ok=True)

        # put the input frames into the new folder
        for frame_name in last_frame_names:
            # do this in a way that overwrites the files and MOVES them
            shutil.move(frame_name, os.path.join(temp_folder, os.path.basename(frame_name)))

        # new input frame names list
        last_frame_names = [os.path.join(temp_folder, os.path.basename(frame_name)) for frame_name in last_frame_names]
        last_frame_names.sort()

        # Create a file with the list of input frame paths in the temporary folder
        input_frame_list_file = os.path.join(temp_folder, "input_frames.txt")
        input_frame_list_file = os.path.abspath(input_frame_list_file)
        with open(input_frame_list_file, 'w') as file:
            # need to do this in a manner that puts the first frame at the top of the file.
            file.write('\n'.join([f"file '{frame_name}'" for frame_name in last_frame_names]))

        # Get the output frame names for the interpolated frames
        # output_frame_names = [video_folder_def.reserve_next_frame() for _ in range(target_n_frames)]
        output_frame_names = [video_folder_def.get_next_file_name(i) for i in range(target_n_frames)]

        # delete the frames on the list
        # for frame_name in output_frame_names:
        #    os.remove(frame_name)

        temp_output_frame_pattern = os.path.join(video_folder_def["video_folder"], "temp",
                                                 "tmp_frame_%010d." + video_folder_def["frame format"].lower())
        temp_output_frame_pattern = os.path.abspath(temp_output_frame_pattern)

        # Create the FFmpeg command for motion interpolation using the frame list file
        ffmpeg_interpolate_cmd = [
            "-f", "concat", "-safe", "0",
            "-i", input_frame_list_file,
            "-vsync", "0", "-filter_complex",
            # f"[0:v]minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=ds:mb_size=16:search_param=8:vsbmc=1[v]",
            # f"[0:v]framerate=fps={target_fps}:interp_start=0:interp_end=255:scene=0[v]",
            # f"[0:v]minterpolate=fps={target_fps}:vsbmc=1[v]",
            f"[0:v]minterpolate=fps={target_fps}:mi_mode=blend[v]",
            "-map", "[v]", "-q:v", "2", "-r", str(target_fps), "-frames", str(target_n_frames),
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
        # remove this without generating an error even if it is not empty
        try:
            os.rmdir(temp_folder)
        except OSError:
            shutil.rmtree(temp_folder)

        return (video_folder_def,)


@ETK_AVF_base
class VideoDefinitionProvider():
    """
    given a VIDEOFOLDER node, and inputs for height, width, and fps, creates a VIDEOFOLDERDEF variable
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder": ("VIDEOFOLDER", {"default": None}),
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
        ret["frame format"] = kwargs["frame format"]
        # a function that gets the frame count from the disk
        ret.get_frame_count = lambda: len([f for f in os.listdir(ret["video_folder"]) if "frame_" in f])
        # a function that gets the next file name and full path dont forget to add 10 zeros to the frame count
        ret.get_next_file_name = lambda x=0: os.path.join(ret['video_folder'],
                                                          f"frame_{str(ret.get_frame_count() + x).zfill(10)}.{ret['frame format'].lower()}")
        ret.get_last_file_name = lambda offset=0: os.path.join(ret['video_folder'],
                                                               f"frame_{str(max(ret.get_frame_count() - offset - 1, 0)).zfill(10)}.{ret['frame format'].lower()}")
        ret.get_current_file_name = lambda: os.path.join(ret['video_folder'],
                                                         f"frame_{str(ret.get_frame_count() - 1).zfill(10)}.{ret['frame format'].lower()}")

        def reserve_next_frame():
            fn = ret.get_next_file_name()
            open(fn, "w").close()
            return fn

        ret.reserve_next_frame = reserve_next_frame

        return (ret,)


@ETK_AVF_base
class VideoFolderProvider():
    """
    given a folder path, creates the folder or empties it entirely returns the folder path as
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

        # check if the folder_path is none
        if kwargs["folder_path"] is None:
            raise ValueError("folder_path must be specified")

        # set folder exists flag by using the os.path.exists function
        folder_exists = os.path.exists(kwargs["folder_path"])

        if not folder_exists:
            os.makedirs(kwargs["folder_path"])
            folder_exists = True

        # set files in folder flag by using the os.listdir function
        files_in_folder = os.listdir(kwargs["folder_path"])

        # set accessible flag by using the os.access function
        accessible = os.access(kwargs["folder_path"], os.W_OK)
        if not accessible:
            raise ValueError("Folder for video files is not accessible")

        if kwargs["clear_folder"]:
            # if the folder exists, delete all files in the folder
            if folder_exists:
                for file in files_in_folder:
                    # use shutil to remove the file
                    fl = os.path.join(kwargs["folder_path"], file)
                    if os.path.isfile(fl):
                        os.remove(fl)
                    # use shutil to remove the entire path
                    check_this_path = file
                    if len(check_this_path) > 3:
                        shutil.rmtree(check_this_path, ignore_errors=True)
                    else:
                        raise ValueError("Folder path is too short")

        return (kwargs["folder_path"],)


@ETK_AVF_base
class AudioFolderProvider():
    """
    given a folder path, creates the folder or empties it entirely returns the folder path as
    a string but as the node type "AUDIOFOLDER"
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

    RETURN_TYPES = ("AUDIOFOLDER",)
    FUNCTION = "audio_folder_provider"

    def audio_folder_provider(self, **kwargs):
        import shutil
        import os

        # check if the folder_path is none
        if kwargs["folder_path"] is None:
            raise ValueError("folder_path must be specified")

        # set folder exists flag by using the os.path.exists function
        folder_exists = os.path.exists(kwargs["folder_path"])

        if not folder_exists:
            os.makedirs(kwargs["folder_path"])
            folder_exists = True

        # set files in folder flag by using the os.listdir function
        files_in_folder = os.listdir(kwargs["folder_path"])

        # set accessible flag by using the os.access function
        accessible = os.access(kwargs["folder_path"], os.W_OK)
        if not accessible:
            raise ValueError("Folder for audio files is not accessible")

        if kwargs["clear_folder"]:
            # if the folder exists, delete all files in the folder
            if folder_exists:
                for file in files_in_folder:
                    # use shutil to remove the file
                    fl = os.path.join(kwargs["folder_path"], file)
                    if os.path.isfile(fl):
                        os.remove(fl)
                    # use shutil to remove the entire path
                    check_this_path = file
                    if len(check_this_path) > 3:
                        shutil.rmtree(check_this_path, ignore_errors=True)
                    else:
                        raise ValueError("Folder path is too short")

        return (kwargs["folder_path"],)


@ETK_AVF_base
class DictToComfyAudio:
    """
    given a dictionary, creates a comfy audio node
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"dict": ("DICT", {"default": {}})}}
        return ret

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "dict_to_comfy_audio"

    def dict_to_comfy_audio(self, **kwargs):
        return (kwargs["dict"],)


@ETK_AVF_base
class LoadAudioFromServer():
    """
    given a path on the server loads the audio file from the server and returns it as an AUDIO node
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"server_path": ("STRING", {"default": ""})}}
        return ret

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio_from_server"

    def load_audio_from_server(self, **kwargs):
        file_on_server = kwargs["server_path"]
        from comfy_extras.nodes_audio import LoadAudio
        f = LoadAudio().load[0]
        r = f(file_on_server)

        return (r,)


@ETK_AVF_base
class AudioDefinitionProvider():
    """
    given a AUDIOFOLDER node, and inputs for sample rate, and channels, creates a AUDIOFOLDERDEF variable
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"audio_folder": ("AUDIOFOLDER", {"default": None}),
                            "sample rate": ("INT", {"default": 44100}),
                            "channels": ("INT", {"default": 2}),
                            "audio format": (["WAV", "MP3"], {"default": "WAV"}),
                            },
               }
        return ret

    RETURN_TYPES = ("AUDIO_FOLDER_DEF",)
    FUNCTION = "audio_definition_provider"

    def get_num_audio_files_that_match(self, audio_folder):
        audio_files = self.get_all_audio_files(audio_folder)
        return len(audio_files)

    def get_all_audio_files(self, audio_folder):
        import os
        audio_files = os.listdir(audio_folder)
        audio_files = sorted([os.path.join(audio_folder, x) for x in audio_files if "audio_" in x])
        return audio_files

    def audio_definition_provider(self, **kwargs):
        from addict import Dict as adict
        import os
        ret = adict()
        ret["audio_folder"] = kwargs["audio_folder"]
        ret["sample rate"] = kwargs["sample rate"]
        ret["channels"] = kwargs["channels"]
        ret["audio format"] = kwargs["audio format"]
        # a function that gets the frame count from the disk
        ret.get_file_count = lambda: self.get_num_audio_files_that_match(ret["audio_folder"])
        # a function that gets the next file name and full path dont forget to zfill
        ret.get_next_file_name = lambda: f"{ret['audio_folder']}{os.sep}audio_{str(ret.get_file_count()).zfill(10)}.{ret['audio format'].lower()}"
        ret.get_last_file_name = lambda: f"{ret['audio_folder']}{os.sep}audio_{str(ret.get_file_count() - 1).zfill(10)}.{ret['audio format'].lower()}"
        ret.get_all_audio_files = lambda: self.get_all_audio_files(ret["audio_folder"])

        return (ret,)


@ETK_AVF_base
class AddAudioToAudioFolder():
    """
    given a AUDIO_FOLDER_DEF node, and a list of audio files, adds the audio files to the audio folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None}),
                            "AUDIO": ("LIST", {"default": None}),
                            "delete after": ("BOOLEAN", {"default": False}),
                            },
               }
        return ret

    RETURN_TYPES = ("AUDIO_FOLDER_DEF",)
    FUNCTION = "add_audio_to_audio_folder"

    def add_audio_to_audio_folder(self, **kwargs):
        import os

        audio_folder_def = kwargs["audio_folder_def"]
        audio_files = kwargs["AUDIO"]
        delete = kwargs["delete after"]

        for audio_file in audio_files:
            # use ffmpeg_exe to convert the audio file to the correct format which is audio_folder_def["audio format"]
            # use the audio_folder_def["audio format"] to determine the output file name
            # use the audio_folder_def["audio folder"] to determine the output folder
            # use the audio_folder_def["sample rate"] to determine the sample rate
            # use the audio_folder_def["channels"] to determine the number of channels

            # get the output file name
            output_file_name = audio_folder_def.get_next_file_name()
            input_file_name = audio_file

            CMD_AS_LIST = [
                "-y", "-i", input_file_name, "-ar", str(audio_folder_def['sample rate']),
                "-ac", str(audio_folder_def['channels']), output_file_name
            ]

            run_ffmpeg_command(CMD_AS_LIST)
            if delete:
                os.remove(input_file_name)

        return (audio_folder_def,)


@ETK_AVF_base
class AddSilenceToAudioFolder():
    """create and add a new audio file that is silence for the specified duration"""

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None}),
                            "duration": ("FLOAT", {"default": 1.0}),
                            },
               }
        return ret

    RETURN_TYPES = ("AUDIO_FOLDER_DEF",)
    FUNCTION = "add_silence_to_audio_folder"

    def add_silence_to_audio_folder(self, **kwargs):
        audio_folder_def = kwargs["audio_folder_def"]
        duration = kwargs["duration"]

        # get the output file name
        output_file_name = audio_folder_def.get_next_file_name()
        sample_rate = audio_folder_def["sample rate"]
        channels = audio_folder_def["channels"]
        fmt = audio_folder_def["audio format"]

        # create the silence audio file

        if fmt == "WAV":
            fmt_part = "pcm_s16le"
        elif fmt == "MP3":
            fmt_part = "mp3"
        else:
            fmt_part = "pcm_s16le"

        # Create the FFmpeg command for generating silence
        CMD_AS_LIST = [
            "-f", "lavfi", "-i", f"aevalsrc=0:sample_rate={sample_rate}:channel_layout={channels},atrim=0:{duration}",
            "-c:a", fmt_part, output_file_name
        ]

        run_ffmpeg_command(CMD_AS_LIST)

        return (audio_folder_def,)


@ETK_AVF_base
class GetAudioFilesLengthsFromAudioFolder():
    """Get the lengths of all audio files in a given audio folder"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None})}}

    RETURN_TYPES = ("AUDIO_FOLDER_DEF", "LIST",)
    RETURN_NAMES = ("audio_folder_def", "[lens(seconds)]",)
    FUNCTION = "get_audio_files_lengths"

    def get_audio_files_lengths(self, audio_folder_def):
        import os
        import torchaudio

        audio_folder = audio_folder_def["audio_folder"]
        audio_files = os.listdir(audio_folder)
        audio_files_lengths = []

        for audio_file in audio_files:
            if audio_file[-3:] not in ["wav", "mp3"]:
                continue
            audio_path = os.path.join(audio_folder, audio_file)
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            audio_files_lengths.append(duration)

        return (audio_folder_def, audio_files_lengths,)


@ETK_AVF_base
class AudioFolderRenderAudio():
    """ Joins all the audio files """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None}),
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = 'join_audio_files'
    CATEGORY = 'ETK/audio'

    def join_audio_files(self, **kwargs):
        import librosa
        import copy
        import soundfile as sf
        import numpy as np

        audio_folder_def = kwargs['audio_folder_def']
        kwargs = copy.deepcopy(kwargs)
        paths = audio_folder_def.get_all_audio_files()
        output_path = kwargs['output_path']

        # Read the first file to get metadata
        audio_data_list = []
        for path in paths:
            audio_data, sr = librosa.load(path, sr=None)
            audio_data_list.append(audio_data)

        # Concatenate all audio data
        joined_audio_data = np.concatenate(audio_data_list)

        # TODO: need to implement writing in the correct format
        # Save the concatenated audio
        sf.write(output_path, joined_audio_data, sr)

        return (output_path,)


@ETK_AVF_base
class AddFramesToVideoFolder():
    """
    given a VIDEO_FOLDER_DEF node, and a list of frames, adds the frames to the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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

            if current_frame.dtype != torch.uint8:  # handle float type frames
                ndarr = current_frame.permute(2, 0, 1).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu",
                                                                                                             torch.uint8).numpy()
            else:  # handle uint8 frames
                ndarr = current_frame.permute(2, 0, 1).permute(1, 2, 0).numpy()
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
        frames_area_all_the_same = torch.all(frames[0] == frames)
        if frames_area_all_the_same and frames.shape[0] > 1:
            # get a list of file names using reservenexframe
            frame_f_names = [video_folder_def.reserve_next_frame() for _ in range(num_frames)]
            # save the first frame
            AddFramesToVideoFolder.save_or_copy_frame(frames[0], previous_frame, frame_f_names[0], None)
            # copy the first frame to the rest of the frames using os by calling the cli
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

        # get the video folder path
        video_folder = kwargs["video_folder_def"]["video_folder"]

        # get the frame format
        frame_format = kwargs["video_folder_def"]["frame format"]

        # get the height
        height = kwargs["video_folder_def"]["height"]

        # get the width
        width = kwargs["video_folder_def"]["width"]

        # get the fps
        fps = kwargs["video_folder_def"]["fps"]

        # get the frames
        frames = kwargs["IMAGE"]

        # get the rescale method
        rescale_method = kwargs.get("Rescale method", None)
        if not rescale_method:
            rescale_method = kwargs.get("Rescale_method", None)

        free_vram = kwargs.get("Release vram?", None)
        if not free_vram:
            free_vram = kwargs.get("Release_vram_", True)

        orig_frames = frames
        # num_files = len(glob(os.path.join(video_folder, f"frame_*.{frame_format.lower()}")))

        # rescale if needed to the correct size
        if height != frames.shape[1] or width != frames.shape[2]:
            # first since transforms.Resize requires a tensor of shape B,C, H, W
            # we need to permute the tensor to B, C, H, W
            frames = frames.permute(0, 3, 1, 2)
            frames = frames.to("cuda")

            # get the right value for interpolation
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
                rescale_method = lambda x: resize(x, (height, width), interpolation=rescale_type)
                frames = rescale_method(frames)
            else:
                rescale_method = lambda x: lanczos_resize(x, (width, height))
                frames = rescale_method(frames)

            # now we need to permute back to B, H, W, C
            frames = frames.permute(0, 2, 3, 1)
            self.process_frames(frames, rescale_method, kwargs["video_folder_def"])
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
            # torch.cuda.empty_cache()
            # gc.collect()

        initial_memory = psutil.virtual_memory().available
        print(f"after adding frames free memory: {initial_memory / (1024 * 1024):.2f} MB")
        # return the VIDEO_FOLDER_DEF
        return (kwargs["video_folder_def"],)


@ETK_AVF_base
class HoldFrameForSecsInVideoFolder():
    """
    given a VIDEO_FOLDER_DEF node, and a list of frames, first uses RepeatFrames to repeat the frames
    for the correct amount and then adds the frames to the video folder using AddFramesToVideoFolder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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
        # this just saves in an async way
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
        vide_folder_ref = kwargs["video_folder_def"]

        video_folder = kwargs["video_folder_def"]["video_folder"]

        # get the frame format
        frame_format = kwargs["video_folder_def"]["frame format"]

        # get the height
        height = kwargs["video_folder_def"]["height"]

        # get the width
        width = kwargs["video_folder_def"]["width"]

        # get the fps
        fps = kwargs["video_folder_def"]["fps"]

        # get the frames
        frames = kwargs["IMAGE"]

        # get the number of frames
        num_frames = frames.shape[0]

        # get the rescale method
        rescale_method = kwargs.get("Rescale method", None)
        if rescale_method is None:
            rescale_method = kwargs.get("Rescale_method", None)

        # get the number of seconds
        seconds = kwargs["seconds"]

        # get the number of frames to hold
        num_frames_to_hold = int(fps * seconds)

        num_frames_in_this_stack = frames.shape[0]

        # get the number of repeats needed
        num_repeats_needed = num_frames_to_hold // num_frames_in_this_stack

        # get the number of frames left over
        num_frames_left_over = num_frames_to_hold % num_frames_in_this_stack

        # write the first set of frames to disk
        self.saver(frames, kwargs)

        # get the name of the last frame
        source_last_frame_name = vide_folder_ref.get_last_file_name()
        # from the last frame find the rest of the frames names/paths
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
        # we can now repeat the frames
        for i, target_frame_name in enumerate(target_frame_names):
            # keep track of what stack frame we are on
            stack_frame_idx = i % num_frames_in_this_stack
            stack_frame_name_to_use = source_frame_names[stack_frame_idx]
            set_todo[stack_frame_name_to_use].append(target_frame_name)

        # now we can repeat the frames
        for frame_name in set_todo:
            src_frame = frame_name
            target_frames = set_todo[src_frame]
            duplicate_file(src_frame, target_frames)

        return (kwargs["video_folder_def"],)


@ETK_AVF_base
class Listed_HoldFramesForSecsInVideoFolder():
    """This just calls HoldFrameForSecsInVideoFolder multiple times for a list of frames and a list of durations"""

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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
        # the plan is to call HoldFrameForSecsInVideoFolder for each frame and duration
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
class VideoFolderDefInfo():
    """
    Returns everything in the VIDEO_FOLDER_DEF, and also returns information about the status of the folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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
class FadeToInVideoFolder():
    """
    given a VIDEO_FOLDER_DEF node, and a stack of frames, fades the frames in and adds them to the video folder
    """

    @classmethod
    def INPUT_TYPES(cls):
        ret = {"required": {"video_folder_def": ("VIDEO_FOLDER_DEF", {"default": None}),
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

        # usee HoldFrameForSecsInVideoFolder to hold the frame after creating and AddFramesToVideoFolder the transition

        transition_length_secs = int(kwargs["transition ms"] * kwargs["video_folder_def"]["fps"]) / 1000
        hold_frame_length_secs = int(kwargs["seconds"] * kwargs["video_folder_def"]["fps"]) - transition_length_secs

        transition_frame_count = int(kwargs["video_folder_def"]["fps"] * transition_length_secs)

        base_frame_fn = kwargs["video_folder_def"].get_last_file_name()
        # load it into a pytorch tensor (B, H, W, C)
        base_frame = Image.open(base_frame_fn)
        base_frame = torch.tensor(base_frame).permute(2, 0, 1).unsqueeze(0).to("cuda")

        transition_stack = torch.zeros((transition_frame_count,
                                        base_frame.shape[1],
                                        base_frame.shape[2],
                                        base_frame.shape[3]),
                                       dtype=torch.float16, device="cuda")

        # fade from base_frame to the first frame in the stack over the transition length
        new_frame = kwargs["IMAGE"][0].to("cuda")
        for i in range(transition_frame_count):
            alpha = i / transition_frame_count
            new_frame = (1 - alpha) * base_frame + alpha * new_frame
            new_frame = new_frame.to("cpu")
            transition_stack[i] = new_frame

        # add to the video folder
        AddFramesToVideoFolder().add_frames_to_video_folder(
            video_folder_def=kwargs["video_folder_def"],
            IMAGE=transition_stack,
            Rescale_method=kwargs["Rescale method"],
            Release_vram_=kwargs["Release vram?"])

        # hold the frame
        HoldFrameForSecsInVideoFolder().hold_frame_for_secs_in_video_folder(
            video_folder_def=kwargs["video_folder_def"],
            IMAGE=kwargs["IMAGE"],
            seconds=hold_frame_length_secs,
            Rescale_method=kwargs["Rescale method"],
            Release_vram_=kwargs["Release vram?"])

        return (kwargs["video_folder_def"],)


def duplicate_file(source_file, target_files):
    """
    Duplicates a given source file into multiple target files using a single CLI command
    in a non-blocking way. The base directory is automatically determined from the
    source file's location.

    Args:
    - source_file (str): Path to the source file to be duplicated.
    - target_files (list of str): A list of target file paths where the source file should be duplicated.

    Returns:
    - None
    """
    import os
    import subprocess

    # Determine the base directory from the source file's directory
    base_dir = os.path.dirname(os.path.abspath(source_file))

    # Change the current working directory to the base directory
    os.chdir(base_dir)

    # Get the source file name (relative to the base directory)
    source_file_name = os.path.basename(source_file)

    # get the relative paths of the target files
    target_files = [os.path.relpath(target_file, base_dir) for target_file in target_files]

    # Determine the shell command based on the operating system
    copy_command = 'copy' if os.name == 'nt' else 'cp'

    # Construct the command string
    command_parts = [f'{copy_command} "{source_file_name}" "{target}"' for target in target_files]
    command = ' && '.join(command_parts)

    command = command.replace("\"", "")
    command = command.replace("\'", "")
    # print(len(command))

    try:
        # Execute the command in a non-blocking way
        process = subprocess.Popen(command, shell=True, stderr=subprocess.DEVNULL)
        process.wait()

    except subprocess.CalledProcessError as e:
        pass
    except ValueError as e:
        pass
    except Exception as e:
        pass

    if "process" not in locals():
        pass
    if "process" not in locals() or process.returncode != 0:
        # If the command is too long, split it into multiple commands
        # split the command into commands that do 64 at a time
        command_parts = [command_parts[i:i + 64] for i in range(0, len(command_parts), 64)]
        processes = []
        for command_part in command_parts:
            command = ' && '.join(command_part)
            process = subprocess.Popen(command, shell=True, stderr=subprocess.DEVNULL)
            processes.append(process)

        for process in processes:
            process.wait()

        return process


def run_ffmpeg_command(args):
    """
    Executes a command using the ffmpeg executable.

    Parameters:
        args (list): List of command-line arguments for ffmpeg.
    """
    import subprocess
    import imageio_ffmpeg as ffmpeg
    if isinstance(args, str):
        args = args.split()
    cmd = [ffmpeg.get_ffmpeg_exe()] + args
    subprocess.run(cmd)
