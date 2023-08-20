import copy

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


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
            with open(fpath, "rb") as f:
                l_bytes.append(f.read())
            l_metadata.append(GetAudioMetaData().get_audio_metadata(path=fpath)[0])

        return (l_fpaths, l_bytes, l_metadata,)


import librosa
import soundfile as sf
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