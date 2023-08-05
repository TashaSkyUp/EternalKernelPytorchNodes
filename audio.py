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
