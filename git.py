import giturlparse

class PublicGitRepo:
    """replaces the text in the given text string with a given other text at some key positions"""

    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "validate_git_repo_url"

    def validate_git_repo_url(self, text: str, replacement, key1, **kwargs):
        """
        >>> PublicGitRepo().validate_git_repo_url("hello world", "universe", "world", git_url="https://github.com/example/repo.git")
        'hello universe'
        """
        git_url = kwargs.get("git_url")
        parsed_url = giturlparse.parse(git_url)
        if not parsed_url.valid:
            raise ValueError("Invalid Git repository URL")

        return (text.replace(key1, replacement),)
