class PublicGitRepo:
    """replaces the text in the given text string with a given other text at some key positions"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "replacement": ("STRING", {"multiline": True}),
                "key1": ("STRING", {"multiline": False}),
                "git_url": ("STRING", {"multiline": False})  # Added git_url input
            }
        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "prompt_template_handler"

    def prompt_template_handler(self, text: str, replacement, key1, git_url):
        """
        >>> PublicGitRepo().prompt_template_handler("hello world", "universe", "world", "https://github.com/example/repo.git")
        'hello universe'
        """
        return (text.replace(key1, replacement),)
