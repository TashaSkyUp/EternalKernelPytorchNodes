from git import Repo, InvalidGitRepositoryError
import os
import copy
import re
from custom_nodes.EternalKernelLiteGraphNodes.shared import ETK_PATH

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def check_for_js_extension():
    # here we check to see that the js extension is installed at ../../web/extensions/etk/ui_modifications.js
    # if it is not, we will copy it there from the ETK_PATH
    # check if the file exists
    import shutil
    check_full_path = os.path.normpath(os.path.join(ETK_PATH, "../../web/extensions/etk/ui_modifications.js"))
    if not os.path.exists(check_full_path):
        # copy the file
        shutil.copy(os.path.join(ETK_PATH, "ui_modifications.js"), check_full_path)


check_for_js_extension()


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
    CATEGORY = 'ETK/git'

    def func(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        repo_url = kwargs.get('repo_url')
        path = kwargs.get('path')
        token = kwargs.get('token')

        # clone a repo into a new directory, using the custom ssh command
        # token actually is the path to the private key
        # so we already know where the private key is
        # we just need to tell git to use it

        cmd = 'ssh -i {} -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no'.format(token)
        repo = Repo.clone_from(repo_url, path, env={'GIT_SSH_COMMAND': cmd})

        # this should not work but it does.
        Repo.clone_from(repo_url, path, env={'GIT_ASKPASS': 'echo', 'GIT_USERNAME': token})

        # linux / not windows ?
        # this actually worked on linux
        # GIT_SSH_COMMAND="ssh -i /root/.ssh/git.private.key" clone https://github.com/gitpython-developers/GitPython.git
        #

        # use the key at /root/.ssh/git.private.key
        # below needs tested
        ssh_cmd = 'ssh -i /root/.ssh/git.private.key -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no'
        repo = Repo(path)
        with repo.git.custom_environment(GIT_SSH_COMMAND=ssh_cmd):
            repo.clone_from(repo_url, path)

        # return the file structure of the repo
        ld = os.listdir(path)
        return (path, "".join(ld), ld,)


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
