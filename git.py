from git import Repo, InvalidGitRepositoryError
import os
import copy
import re

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def ETK_git_base(cls):
    cls.FUNCTION = "func"
    cls.CATEGORY = "ETK/git"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


@ETK_git_base
class PublicGitRepo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_url": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("bool",)
    FUNCTION = 'validate_git_repo_url'
    CATEGORY = 'ETK/git'


def validate_git_repo_url(self, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    repo_url = kwargs.get('repo_url')
    token = kwargs.get('token')

    # Regular expression to match GitHub repository URLs
    pattern = r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?'
    match = re.match(pattern, repo_url)

    # The URL is valid if there's a match and it covers the whole URL
    return match is not None and match.group() == repo_url


@ETK_git_base
class CloneRepoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_url": ("STRING", {"multiline": False, "default": ""}),
                "path": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LIST",)
    RETURN_NAMES = ("path_passthrough", "file_structure", "file_list",)
    FUNCTION = 'clone_repo'
    CATEGORY = 'ETK/git'

    def clone_repo(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        repo_url = kwargs.get('repo_url')
        path = kwargs.get('path')
        token = kwargs.get('token')
        Repo.clone_from(repo_url, path, env={'GIT_ASKPASS': 'echo', 'GIT_USERNAME': token})

        # return the file structure of the repo
        ld = os.listdir(path)
        return ("".join(ld), ld,)


@ETK_git_base
class PullRepoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_path": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = 'pull_repo'
    CATEGORY = 'ETK/git'


def pull_repo(self, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    repo_path = kwargs.get('repo_path')
    token = kwargs.get('token')
    repo = Repo(repo_path)
    origin = repo.remotes.origin
    # with repo.git(GIT_ASKPASS='echo', GIT_USERNAME=token):
    origin.pull()

    # with repo.git.custom_environment(GIT_ASKPASS='echo', GIT_USERNAME=token):
    #    origin.pull()


@ETK_git_base
class PushRepoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_path": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = 'push_repo'
    CATEGORY = 'ETK/git'


def push_repo(self, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    repo_path = kwargs.get('repo_path')
    token = kwargs.get('token')
    repo = Repo(repo_path)
    origin = repo.remotes.origin
    # with repo.git.custom_environment(GIT_ASKPASS='echo', GIT_USERNAME=token):
    origin.push()


@ETK_git_base
class SendCommitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_path": ("STRING", {"multiline": False, "default": ""}),
                "commit_message": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = 'send_commit'
    CATEGORY = 'ETK/git'


def send_commit(self, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    repo_path = kwargs.get('repo_path')
    commit_message = kwargs.get('commit_message')
    repo = Repo(repo_path)
    repo.git.add(A=True)
    repo.index.commit(commit_message)


@ETK_git_base
class CreateRepoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": ""}),
                "token": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = 'create_repo'
    CATEGORY = 'ETK/git'


def create_repo(self, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    path = kwargs.get('path')
    Repo.init(path)
