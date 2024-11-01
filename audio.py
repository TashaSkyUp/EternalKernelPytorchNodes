import shutil

import copy
import imageio_ffmpeg as ffmpeg
import joblib
from .config import config_settings

temp_dir = config_settings["tmp_dir"]
from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


# NODE_CLASS_MAPPINGS = {}
# NODE_DISPLAY_NAME_MAPPINGS = {}


# Create a memory object for caching
# memory = joblib.Memory(temp_dir, verbose=0)

def ETK_audio_base(cls):
    cls.CATEGORY = "ETK/audio"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


@ETK_audio_base
class GetAudioMetaData:
    """ Returns a dictionary of audio metadata """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": ""}),
            }

        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = 'get_audio_metadata'
    CATEGORY = 'ETK/audio'

    def get_audio_metadata(self, **kwargs):
        import audioread
        kwargs = copy.deepcopy(kwargs)
        path = kwargs.get('path')
        metadata = {}
        with audioread.audio_open(path) as f:
            metadata['duration'] = f.duration
            metadata['samplerate'] = f.samplerate
            metadata['channels'] = f.channels

        return (metadata,)


@ETK_audio_base
class AudioFileListFromPathPattern:
    """ Returns a list of audio files from a path pattern """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": ""}),
            }

        }

    RETURN_TYPES = ("LIST", "LIST", "LIST",)
    RETURN_NAMES = ("list_of_fpaths", "list_of_bytes", "list_of_metadata")
    FUNCTION = 'get_audio_file_list_from_path_pattern'
    CATEGORY = 'ETK/audio'

    def get_audio_file_list_from_path_pattern(self, **kwargs):
        import glob
        kwargs = copy.deepcopy(kwargs)
        path = kwargs.get('path')
        l_fpaths = glob.glob(path)
        l_bytes = []
        l_metadata = []
        for fpath in l_fpaths:
            if fpath[-4:] in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".alac", ".au", ".raw",
                              ".mp2", ]:
                with open(fpath, "rb") as f:
                    l_bytes.append(f.read())
                l_metadata.append(GetAudioMetaData().get_audio_metadata(path=fpath)[0])

        return (l_fpaths, l_bytes, l_metadata,)


import numpy as np


@ETK_audio_base
class JoinAudioFiles:
    """ Joins a list of audio files into a single audio file """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paths": ("LIST", {"default": []}),
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "BYTES", "DICT",)
    FUNCTION = 'join_audio_files'
    CATEGORY = 'ETK/audio'

    def join_audio_files(self, **kwargs):
        import librosa
        import soundfile as sf
        kwargs = copy.deepcopy(kwargs)
        paths = kwargs['paths']
        output_path = kwargs['output_path']

        # Read the first file to get metadata
        audio_data_list = []
        for path in paths:
            audio_data, sr = librosa.load(path, sr=None)
            audio_data_list.append(audio_data)

        # Concatenate all audio data
        joined_audio_data = np.concatenate(audio_data_list)

        # Save the concatenated audio
        sf.write(output_path, joined_audio_data, sr)

        # Get metadata
        metadata = {
            'duration': librosa.get_duration(y=joined_audio_data, sr=sr),
            'samplerate': sr
        }

        # Get bytes
        with open(output_path, "rb") as f:
            file_bytes = f.read()

        return (output_path, file_bytes, metadata,)


@ETK_audio_base
class ExtractAudioFromVideo:
    """ Extracts audio from a video file and saves it as an audio file with a specified format and bitrate """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False, "default": ""}),
                "format": (["wav", "mp3"], {"default": "wav"}),
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "bitrate (MP3)": ("INT", {"default": 192})  # Default bitrate value (change as needed)
            }
        }

    RETURN_TYPES = ("STRING", "BYTES")
    RETURN_NAMES = ("output_wave_file_path", "audio_bytes")
    FUNCTION = 'extract_audio_from_video'
    CATEGORY = 'ETK/audio'

    def extract_audio_from_video(self, **kwargs):
        import moviepy.editor as mp
        import io

        kwargs = copy.deepcopy(kwargs)
        video_path = kwargs.get('video_path')
        output_path = kwargs.get('output_path', "tmp.wav")
        audio_format = kwargs.get('format', 'wav')
        bitrate = kwargs.get('bitrate', None)  # Default bitrate is None

        # Load the video clip
        video_clip = mp.VideoFileClip(video_path)

        # Determine the appropriate codec based on the audio format
        if audio_format == 'wav':
            codec = "pcm_s16le"  # For WAV format
        elif audio_format == 'mp3':
            codec = "mp3"  # For MP3 format

        # Extract audio and save it with the specified format and bitrate
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_path, codec=codec, bitrate=bitrate)

        # Read the generated audio file as bytes
        audio_bytes = None
        with open(output_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()

        return (output_path, audio_bytes,)


# import torchaudio
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
#
# model = MusicGen.get_pretrained('melody')
# model.set_generation_params(duration=8)  # generate 8 seconds.
#
# descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
#
# melody, sr = torchaudio.load('./assets/bach.mp3')
# # generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
#
# for idx, one_wav in enumerate(wav):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

@ETK_audio_base
class GenerateMusic:
    """ Generates music from a given melody and descriptions """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "description": ("STRING", {"default": ""}),
                "[description]": ("LIST", {"default": None}),
            },
            "required": {
                "model_name": (["facebook/musicgen-melody-large",
                                "facebook/musicgen-melody-small",
                                "facebook/musicgen-melody-medium",

                                "facebook/musicgen-stereo-melody-large",
                                "facebook/musicgen-stereo-melody-small",
                                "facebook/musicgen-stereo-melody-medium",

                                "facebook/musicgen-large",
                                "facebook/musicgen-small",
                                "facebook/musicgen-medium",

                                "facebook/musicgen-stereo-large",
                                "facebook/musicgen-stereo-small",
                                "facebook/musicgen-stereo-medium",

                                ],),
                "duration": ("INT", {"default": None}),
                "[duration]": ("LIST", {"default": None}),
                "file": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST",)
    RETURN_NAMES = ("file", "files_list",)
    FUNCTION = 'generate_music'
    CATEGORY = 'ETK/audio'

    def create_command(self, kwargs):
        import os
        musicgencli_path = os.path.join(os.path.dirname(__file__), "modules", "musicgencli.py")
        ukwargs = copy.deepcopy(kwargs)
        nl = "\n"
        ukwargs["description"] = kwargs.get("[description]", [])
        ukwargs.pop("[description]")
        ukwargs["duration"] = kwargs.get("[duration]", [])
        ukwargs.pop("[duration]")
        ukwargs["description"] = str([i.replace("'", "") for i in ukwargs["description"]])
        command = " ".join([f"--{k} \"{str(v).replace(nl, ' ')}\"" for k, v in ukwargs.items()])
        command = f"python {musicgencli_path} {command}"
        return command

    def generate_music(self, **kwargs):
        # use the cli to call the cli music generator at ./modules/musicgencli.py
        import os
        import subprocess

        command = self.create_command(kwargs)

        # Execute the command caputring the output
        try:
            output = subprocess.check_output(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        # use the os to find the new files
        new_files = []
        for file in os.listdir(os.path.dirname(kwargs['file'])):
            if file.endswith(".wav"):
                new_files.append(file)

        return (kwargs['file'], new_files,)

    def generate_music_old(self, **kwargs):
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        import torch
        torch.cuda.empty_cache()

        import random

        # Set seed, -1 means random seed
        seed = kwargs.get('seed', -1)
        if kwargs.get('seed') == -1:
            seed = random.randint(0, 100000)
        old_seed_values = set_all_seeds(seed)

        kwargs = copy.deepcopy(kwargs)
        description = kwargs.get('description')
        descriptions = kwargs.get('[description]')

        if description and not descriptions:
            descriptions = [description]

        file = kwargs.get('file')
        # add .wave if not there
        if not file.endswith('.wav'):
            file = file + ".wav"

        model_name = kwargs.get('model_name', 'melody')
        duration = kwargs.get('duration', 10)
        durations = kwargs.get('[duration]', None)

        if duration:
            durations = [duration]

        model = MusicGen.get_pretrained(model_name)
        o_files = []
        for i, description in enumerate(descriptions):
            ii = str(i).zfill(1 + len(str(1 + len(descriptions))))
            file_to_use = file.replace(".wav", f"_{ii}.wav")
            duration_to_use = durations[i]

            model.set_generation_params(duration=duration_to_use)
            wav = model.generate([description])
            # q = MusicGen.set_generation_params()
            audio_write(file_to_use.replace(".wav", ""), wav[0].cpu(), model.sample_rate, strategy="loudness")
            o_files.append(file_to_use)

        # Reset seed
        reset_seeds(old_seed_values)
        torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        return (o_files[0], o_files,)


def set_all_seeds(seed):
    import random
    import os
    import numpy as np
    import torch
    old_values = {}
    old_values['random'] = random.getstate()
    old_values['numpy'] = np.random.get_state()
    old_values['torch'] = torch.get_rng_state()
    old_values['torch_cuda'] = torch.cuda.get_rng_state()
    old_values['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    old_values['torch_deterministic'] = torch.backends.cudnn.deterministic

    # From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return old_values


def reset_seeds(old_values):
    import random
    import numpy as np
    import torch

    random.setstate(old_values['random'])
    np.random.set_state(old_values['numpy'])
    torch.set_rng_state(old_values['torch'])
    torch.cuda.set_rng_state(old_values['torch_cuda'])
    torch.cuda.set_rng_state_all(old_values['torch_cuda_all'])
    torch.backends.cudnn.deterministic = old_values['torch_deterministic']


@ETK_audio_base
class XttsNode:
    """ Converts text to speech """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": ("STRING", {"default": "cuda"}),
            },
            "optional": {
                "text": ("STRING", {"multiline": False, "default": ""}),
                "[text]": ("LIST", {"multiline": False, "default": ""}),
                "speaker_wav": ("STRING", {"multiline": False, "default": ""}),
                "language_code": (
                    ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja',
                     'hi'], {"default": "en"}),
                "output_path": ("STRING", {"multiline": False, "default": ""}),
                "AUDIO_FOLDER_DEF": ("AUDIO_FOLDER_DEF", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST", "STRING", "AUDIO_FOLDER_DEF",)
    RETURN_NAMES = ("output file", "output files list", "output folder", "AUDIO_FOLDER_DEF",)
    FUNCTION = 'xtts'
    CATEGORY = 'ETK/audio'

    def xtts(self=None, **kwargs):
        import os
        import subprocess
        import sys
        import shutil
        import tempfile
        import uuid
        from .config import this_file_path

        # Handle audio_folder_def
        audio_folder_def = kwargs.get("AUDIO_FOLDER_DEF", None)

        # Get user-specified output path
        base_output_path = kwargs.get('output_path', '.')

        # Generate a random temporary folder using tempfile.mkdtemp (persistent until explicitly removed)
        temp_folder = tempfile.mkdtemp(dir=base_output_path)

        python_exe = sys.executable

        # Path to the xttscli.py script
        xttscli_path = os.path.join(this_file_path, "modules", "xttscli.py")

        # Check if '[text]' is in kwargs and if it is a list
        if '[text]' in kwargs and isinstance(kwargs['[text]'], list):
            texts = kwargs['[text]']
        else:
            texts = [kwargs.get('text', "")]

        # Create a temporary file in the given folder with the speech separated by new lines
        fpn = os.path.join(temp_folder, "temp.txt")
        with open(fpn, "w") as f:
            for text in texts:
                f.write(text + "\n")

        # Create the command
        command = [
            python_exe, xttscli_path,
            "--file", fpn,
            "--folder", temp_folder,
            "--lang", kwargs.get('lang', 'en'),
            "--speaker", kwargs.get('speaker_wav', ''),
            "--device", kwargs.get('device', 'cuda:0'),
        ]

        # Write the command to "temp_command.txt" in the same folder as this file is in
        with open(os.path.join(this_file_path, "temp_command.txt"), "w") as f:
            f.write(" ".join(command))

        # Execute the command in its own full process and receive the output
        try:
            # Capture both stdout and stderr
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            generated = output.decode("utf-8")

            if "Generated files: (" in generated:
                # Extract generated files
                generated = str(generated)
                generated = generated.split(r"Generated files: (")[1]
                generated_files = generated.split(")")[0]
                generated_files = eval(generated_files)

                # Get the output folder
                output_folder = os.path.dirname(kwargs.get('file', "output.wav"))

                # Move the files to the correct location
                out_files = []
                for i, src_file in enumerate(generated_files):
                    if audio_folder_def:
                        target_file = audio_folder_def.get_next_file_name()
                    else:
                        target_file = os.path.join(temp_folder, f"output_{i}.wav")
                    shutil.copy(src_file, target_file)
                    out_files.append(target_file)

            else:
                raise Exception(f"Error with xttscli, output was: {generated}")

        except subprocess.CalledProcessError as e:
            # Output the captured stdout + stderr
            print(f"Command failed with error: {e.output.decode('utf-8')}")
            raise e

        # Return values in a consistent manner
        return (
            out_files[0],
            out_files,
            temp_folder,
            audio_folder_def if audio_folder_def else None
        )


import moviepy.editor as mpy


class MovieCrossfadeMultiAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path1": ("STRING", {}),
                "video_path2": ("STRING", {}),
                "crossfade_duration": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "crossfade_videos"
    CATEGORY = "video_processing"

    def crossfade_videos(self, video_path1, video_path2, crossfade_duration):
        # Load video clips
        clip1 = mpy.VideoFileClip(video_path1)
        clip2 = mpy.VideoFileClip(video_path2)

        # Extract primary audio tracks (assuming the first track is primary)
        audio1 = clip1.audio
        audio2 = clip2.audio

        # Crossfade audio tracks
        audio_crossfade = mpy.CompositeAudioClip([audio1, audio2.set_start(clip1.duration - crossfade_duration)])

        # Crossfade video clips
        video_crossfade = mpy.concatenate_videoclips([clip1, clip2.set_start(clip1.duration - crossfade_duration)],
                                                     method="compose", padding=-crossfade_duration)

        # Set the crossfaded audio to the crossfaded video
        final_clip = video_crossfade.set_audio(audio_crossfade)

        # Output attributes
        self.video = final_clip

        return {
            "video": self.video
        }


@ETK_audio_base
class EtkSaveAudio:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",),
                             "filename": ("STRING", {"default": ""})},
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    def save_audio(self, audio, filename=""):
        results = list()

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            file = f"{filename}_{batch_number:05}.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(file, 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "type": self.type
            })

        return {"ui": {"audio": results}, "result": (filename,)}


@ETK_audio_base
class AdjustAudioVolumeForFiles():
    """Adjusts the volume of a list of audio files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_files": ("LIST", {"default": []}),
                "volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "output_folder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("modified_audio_paths",)
    CATEGORY = "audio_processing"
    OUTPUT_NODE = True
    FUNCTION = "handler"

    def handler(self, **kwargs):
        import os
        from os.path import splitext, join
        audio_files = kwargs["audio_files"]
        volume = kwargs["volume"]
        output_folder = kwargs.get("output_folder", "")

        modified_audio_paths = []

        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError("FFmpeg could not be found.")

        for audio_file in audio_files:
            output_file = splitext(audio_file)[0] + f"_adjusted_volume.mp3"
            if output_folder:
                output_file = join(output_folder, os.path.basename(output_file))

            command = [
                ffmpeg_exe, '-i', audio_file,
                '-filter:a', f'volume={volume}',
                '-y', output_file
            ]

            cmd = ' '.join(command)
            print(cmd)
            os.system(cmd)

            modified_audio_paths.append(output_file)

        return (modified_audio_paths,)


if __name__ == '__main__':
    # do some testing
    wave_file = """C:\\Users\\Tasha\\Nextcloud\\ETK\\YT\\Monsters\\The Haunter of the Ring\\The Haunter of the Ring real.wav"""
    xtts_node = XttsNode()
    xtts_node.xtts(speaker_wav=wave_file,
                   text="Hello world! I'm alive, you had better believe it! muahahaha ---- I'm alive! HAHAHAHA! hahahahahahahaha",
                   language_code="en",
                   output_path="output.wav")
    # use windows to play the file
    import os

    os.system("start output.wav")

    # Paths to the video files to crossfade
    video_path1 = 'path/to/your/first/video.mp4'
    video_path2 = 'path/to/your/second/video.mp4'

    # Duration of the crossfade in seconds
    crossfade_duration = 2.0  # Example duration

    # Create an instance of the MovieCrossfadeMultiAudio node
    crossfade_node = MovieCrossfadeMultiAudio()

    # Execute the node's crossfade function
    output_video = crossfade_node.crossfade_videos(video_path1, video_path2, crossfade_duration)

    # Save the output video to a file
    output_video_path = 'path/to/your/output/crossfaded_video.mp4'
    output_video['video'].write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    print(f'Crossfaded video saved to {output_video_path}')
