import subprocess
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def ETK_youtube_base(cls):
    cls.CATEGORY = "ETK/Youtube"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


@ETK_youtube_base
class YoutubeUploadNode:
    VALID_PRIVACY_STATUSES = ["public", "private", "unlisted"]
    VALID_CATEGORIES = [1, 2, 10, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]
    CATEGORY_NAMES = ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports', 'Travel & Events',
                      'Gaming', 'People & Blogs', 'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style',
                      'Education', 'Science & Technology', 'Nonprofits & Activism']

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"description": "Path to the video file to upload"}),
                "category": (cls.CATEGORY_NAMES,),
                "title": ("STRING", {"default": "My video", "description": "Title of the video"}),
                "description": ("STRING", {"default": "My video description", "multiline": True,
                                           "description": "Description of the video"}),
                "keywords": ("STRING", {"default": "tag1, tag2, tag3", "description": "Keywords for the video"}),
                "privacy_status": (cls.VALID_PRIVACY_STATUSES,)
            },
            "optional": {
                "thumbnail_path": ("STRING", {"default": "", "description": "Path to the thumbnail image file"})
            }
        }

    RETURN_TYPES = ('STRING',)
    FUNCTION = "upload_video"
    CATEGORY = "ETK/Youtube"

    def upload_video(self, file_path, category, title, description, keywords, privacy_status, thumbnail_path=""):
        import sys
        my_python_exe = sys.executable

        category_idx = self.CATEGORY_NAMES.index(category)
        category = self.VALID_CATEGORIES[category_idx]

        args = ["--file", file_path, "--category", str(category), "--title", title, "--description", description,
                "--keywords", keywords, "--privacyStatus", privacy_status]

        if thumbnail_path:
            args.extend(["--thumbnail", thumbnail_path])

        current_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        result = subprocess.run([my_python_exe, "./modules/youtube.py"] + args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)

        os.chdir(current_dir)
        error = result.stderr
        if error:
            raise Exception(error)

        return (result.stdout,)


if __name__ == "__main__":
    pass
