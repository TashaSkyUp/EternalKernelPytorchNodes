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

    def validate_git_repo_url(self, text: str, replacement, key1, git_url):
        """
        >>> PublicGitRepo().validate_git_repo_url("hello world", "universe", "world", "https://github.com/example/repo.git")
        'hello universe'
        """
        parsed_url = giturlparse.parse(git_url)
        if not parsed_url.valid:
            raise ValueError("Invalid Git repository URL")

        return (text.replace(key1, replacement),)
