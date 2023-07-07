import unittest
from unittest.mock import patch, Mock
from git import Repo
from git.exc import GitCommandError
import os

os.putenv('UNIT_TEST', 'True')
from ..git import PublicGitRepo, CloneRepoNode, PullRepoNode, PushRepoNode, SendCommitNode, CreateRepoNode

repo_path = './tmp/test_repo'
commit_message = 'message'
token = 'token'
# use tashaskyups's repo on github for testing
repo_url = "https://github.com/TashaSkyUp/public_repo.git"


class BaseTestCase(unittest.TestCase):
    pass


class TestPublicGitRepo(BaseTestCase):
    def test_validate_git_repo_url(self):
        repo = PublicGitRepo()
        # Test with a valid URL
        result = repo.validate_git_repo_url(repo_url=repo_url, token=token)
        self.assertTrue(result)

        # Test with an invalid URL
        result = repo.validate_git_repo_url(repo_url='https://github.com', token=token)
        self.assertFalse(result)


class TestCloneRepoNode(BaseTestCase):
    # @patch.object(Repo, 'clone_from')
    def test_clone_repo(self):
        import shutil
        # delete the repo if it exists first
        if os.path.exists(repo_path):
            # this fails so
            #    shutil.rmtree(repo_path)
            # use this instead
            os.system(f'rmdir "{repo_path}" /S /Q ')

        node = CloneRepoNode()
        node.clone_repo(repo_url=repo_url, path=repo_path, token=token)
        # shutil.rmtree(repo_path)
        # os.system(f'rmdir "{repo_path}" /S /Q ')
        # mock_clone_from.assert_called_once_with(repo_url, repo_path,
        #                                        env={'GIT_ASKPASS': 'echo', 'GIT_USERNAME': 'token'})


class TestPullRepoNode(BaseTestCase):
    # @patch.object(Repo, 'remotes')
    # @patch.object(Repo, '__init__', return_value=None)
    def test_pull_repo(self):
        # mock_origin = Mock()
        # mock_remotes.origin = mock_origin
        node = PullRepoNode()
        node.pull_repo(repo_path=repo_path, token=token)
        # mock_origin.pull.assert_called_once()


class TestSendCommitNode(BaseTestCase):
    # @patch.object(Repo, 'git')
    # @patch.object(Repo, 'index')
    # @patch.object(Repo, '__init__', return_value=None)
    def test_send_commit(self):
        import hashlib
        data = hashlib.md5().hexdigest()
        # write to a file
        with open(f'{repo_path}/test.txt', 'w') as f:
            f.write(data)
        # now commit the file
        node = SendCommitNode()
        node.send_commit(repo_path=repo_path, commit_message=commit_message, token=token)
        # mock_git.add.assert_called_once_with(A=True)
        # mock_index.commit.assert_called_once_with(commit_message)


class TestPushRepoNode(BaseTestCase):
    # @patch.object(Repo, 'remotes')
    # @patch.object(Repo, '__init__', return_value=None)
    def test_push_repo(self):
        # mock_origin = Mock()
        # mock_remotes.origin = mock_origin
        node = PushRepoNode()
        node.push_repo(repo_path=repo_path, token=token)
        # mock_origin.push.assert_called_once()


class TestCreateRepoNode(BaseTestCase):
    @patch.object(Repo, 'init')
    def test_create_repo(self, mock_init):
        node = CreateRepoNode()
        node.create_repo(path=repo_path, token=token)
        mock_init.assert_called_once_with(repo_path)


if __name__ == '__main__':
    # run the tests one at a time in a n order that makes sense
    runner = unittest.TextTestRunner()

    suite = unittest.TestSuite()
    suite.addTest(TestCloneRepoNode('test_clone_repo'))
    runner.run(suite)

    suite = unittest.TestSuite()
    suite.addTest(TestSendCommitNode('test_send_commit'))
    runner.run(suite)

    suite = unittest.TestSuite()
    suite.addTest(TestPushRepoNode('test_push_repo'))
    runner.run(suite)

    # suite.addTest(TestPublicGitRepo('test_validate_git_repo_url'))
    # suite.addTest(TestCreateRepoNode('test_create_repo'))
    # suite.addTest(TestPullRepoNode('test_pull_repo'))

    # run all the tests
    # unittest.main()
