import giturlparse

class PublicGitRepo:
    """replaces the text in the given text string with a given other text at some key positions"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),  # Added url input
                "email": ("STRING", {"multiline": False})  # Added email input
            }
        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "validate_git_repo_url"

    def validate_git_repo_url(self,**kwargs):
        git_url = kwargs.get("url", None)
        parsed_url = giturlparse.parse(git_url)

        return (git_url,)
