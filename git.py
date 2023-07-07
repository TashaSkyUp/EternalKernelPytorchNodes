import giturlparse
import git

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

    CATEGORY = "ETK/git"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "validate_git_repo_url"

    def validate_git_repo_url(self, **kwargs):
        git_url = kwargs.get("url", None)
        parsed_url = giturlparse.parse(git_url)

        return (git_url,)

class SendCommitNode:
    """Sends a commit to a public Git repository"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_path": ("STRING", {"multiline": False}),  # Added repo_path input
                "commit_message": ("STRING", {"multiline": False})  # Added commit_message input
            }
        }

    CATEGORY = "ETK/git"
    RETURN_TYPES = ("NONE",)
    FUNCTION = "send_commit"

    def send_commit(self, **kwargs):
        repo_path = kwargs.get("repo_path", None)
        commit_message = kwargs.get("commit_message", None)

        repo = git.Repo(repo_path)
        repo.git.add(all=True)
        repo.index.commit(commit_message)
        repo.remotes.origin.push()
