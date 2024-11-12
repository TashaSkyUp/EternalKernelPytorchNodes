# common.py
from .. import NODE_CLASS_MAPPINGS
from .. import NODE_DISPLAY_NAME_MAPPINGS


def ETK_AVF_base(cls):
  cls.CATEGORY = "ETK/AVF"
  # Add spaces to the camel case class name
  pretty_name = cls.__name__
  for i in range(1, len(pretty_name)):
      if pretty_name[i].isupper():
          pretty_name = pretty_name[:i] + " " + pretty_name[i:]
  cls.DISPLAY_NAME = pretty_name
  NODE_CLASS_MAPPINGS[cls.__name__] = cls
  NODE_DISPLAY_NAME_MAPPINGS[cls.DISPLAY_NAME] = pretty_name

  return cls


import threading


def duplicate_file(src_file, dest_files):
    import shutil
    threads = []
    for dest_file in dest_files:
        thread = threading.Thread(target=shutil.copy2, args=(src_file, dest_file))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

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