# audio_module.py
from .common import ETK_AVF_base, duplicate_file, run_ffmpeg_command


@ETK_AVF_base
class AudioFolderProvider:
  """
  Given a folder path, creates the folder or empties it entirely returns the folder path as
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
  Given a dictionary, creates a comfy audio node
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
class LoadAudioFromServer:
  """
  Given a path on the server loads the audio file from the server and returns it as an AUDIO node
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
      f = LoadAudio().load
      r = f(file_on_server)[0]

      return (r,)


@ETK_AVF_base
class AudioDefinitionProvider:
  """
  Given a AUDIOFOLDER node, and inputs for sample rate, and channels, creates a AUDIOFOLDERDEF variable
  """

  @classmethod
  def INPUT_TYPES(cls):
      ret = {
          "required": {
              "audio_folder": ("AUDIOFOLDER", {"default": None}),
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
      ret["sample rate"] = kwargs.get("sample rate") or kwargs.get("sample_rate")
      ret["audio format"] = kwargs.get("audio format") or kwargs.get("audio_format")
      ret["channels"] = kwargs.get("channels")
      # a function that gets the frame count from the disk
      ret.get_file_count = lambda: self.get_num_audio_files_that_match(ret["audio_folder"])
      # a function that gets the next file name and full path dont forget to zfill
      ret.get_next_file_name = lambda: f"{ret['audio_folder']}{os.sep}audio_{str(ret.get_file_count()).zfill(10)}.{ret['audio format'].lower()}"
      ret.get_last_file_name = lambda: f"{ret['audio_folder']}{os.sep}audio_{str(ret.get_file_count() - 1).zfill(10)}.{ret['audio format'].lower()}"
      ret.get_all_audio_files = lambda: self.get_all_audio_files(ret["audio_folder"])

      return (ret,)


@ETK_AVF_base
class AddAudioToAudioFolder:
  """
  Given a AUDIO_FOLDER_DEF node, and a list of audio files, adds the audio files to the audio folder
  """

  @classmethod
  def INPUT_TYPES(cls):
      ret = {
          "required": {
              "audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None}),
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
              "-y",
              "-i",
              input_file_name,
              "-ar",
              str(audio_folder_def['sample rate']),
              "-ac",
              str(audio_folder_def['channels']),
              output_file_name
          ]

          run_ffmpeg_command(CMD_AS_LIST)
          if delete:
              os.remove(input_file_name)

      return (audio_folder_def,)


@ETK_AVF_base
class AddSilenceToAudioFolder:
  """Create and add a new audio file that is silence for the specified duration"""

  @classmethod
  def INPUT_TYPES(cls):
      ret = {
          "required": {
              "audio_folder_def": ("AUDIO_FOLDER_DEF", {"default": None}),
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
          "-f",
          "lavfi",
          "-i",
          f"aevalsrc=0:sample_rate={sample_rate}:channel_layout={channels},atrim=0:{duration}",
          "-c:a",
          fmt_part,
          output_file_name
      ]

      run_ffmpeg_command(CMD_AS_LIST)

      return (audio_folder_def,)


@ETK_AVF_base
class GetAudioFilesLengthsFromAudioFolder:
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
          if audio_file[-3:].lower() not in ["wav", "mp3"]:
              continue
          audio_path = os.path.join(audio_folder, audio_file)
          waveform, sample_rate = torchaudio.load(audio_path)
          duration = waveform.shape[1] / sample_rate
          audio_files_lengths.append(duration)

      return (audio_folder_def, audio_files_lengths,)


@ETK_AVF_base
class AudioFolderRenderAudio:
  """Joins all the audio files"""

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
  FUNCTION = "join_audio_files"
  CATEGORY = "ETK/audio"

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