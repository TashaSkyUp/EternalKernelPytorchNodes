import json
from server import PromptServer
from aiohttp import web
from importlib import reload
import logging
import os
from folder_paths import output_directory, input_directory, temp_directory
from custom_nodes.EternalKernelLiteGraphNodes.shared import GDE_PATH
import custom_nodes.EternalKernelLiteGraphNodes.shared as shared

fake_git = None
real_git_url = "https://github.com/story-squad/GDE_Graph_IO.git"  # is this actually an uri?


class GitIO:
    def __init__(self, i_real_git_url: str, i_gde_path: str):
        self.gde_path = i_gde_path
        self.real_git_io_path = self.setup_git_io(i_real_git_url)

    def setup_git_io(self, l_real_git_url: str):
        real_git_path = os.path.join(self.gde_path, "data")
        real_git_path = os.path.join(real_git_path, "GDE_Graph_IO")

        # check to see if it exists and is a git repo, if not then clone it
        if os.path.exists(real_git_path):
            # check to see if it is a git repo
            if os.path.exists(os.path.join(real_git_path, ".git")):
                print("git io found")
            else:
                # not a git repo, delete it and clone it
                print("git io error, not a git repo")
        else:
            # clone it
            print("git io not found")
            print("cloning git io repo")
            os.system("git clone " + l_real_git_url + " " + real_git_path)

        return real_git_path

    def ensure_git_state(self, user: str):
        """works in windows or linux, ensures that the git repo is in the correct state"""
        if not os.path.exists(self.real_git_io_path):
            raise Exception("git io path not found")
        if not os.path.exists(os.path.join(self.real_git_io_path, ".git")):
            raise Exception("git io error, not a git repo")
        # get the current branch
        current_branch = os.popen("git branch --show-current").read()
        # if the current branch is not the user's branch, then switch to it
        if current_branch != user:
            # also check if the branch exists, if not then create it
            branches = os.popen("git branch").read()
            if user not in branches:
                os.system("git branch " + user)
                os.system("git push --set-upstream origin " + user)
            # switch to the user's branch
            os.system("git checkout " + user)

        # pull the latest from the user's branch
        os.system("git pull")

    def git_save_version_1(self, user: str, file_name: str, file_data: str):
        """
        for git save version 1 each user will have their own branch,
        and the data will be saved to a file in the user's branch
        version > 1 may include other strategies
        """
        self.ensure_git_state(user)
        # change to correct directory
        os.chdir(self.real_git_io_path)
        # switch to the users branch
        os.system("git checkout " + user)
        # save the file data as the file to the disk
        with open(os.path.join(self.real_git_io_path, file_name), "w", encoding="utf-8") as f:
            f.write(file_data)
        # add the file to the git repo
        os.system("git add " + file_name)
        # commit the file to the git repo
        os.system("git commit -m \"saving file " + file_name + "\"")
        # push the file to the git repo
        os.system("git push")

    def git_load_version_1(self, user_name, file_name):
        """
        for git load version 1 each user will have their own branch,
        and the data will be loaded from a file in the user's branch
        version > 1 may include other strategies
        """
        self.ensure_git_state(user_name)
        # change to correct directory
        os.chdir(self.real_git_io_path)
        # switch to the users branch
        os.system("git checkout " + user_name)
        # pull the latest from the user's branch
        os.system("git pull")
        # read the file from the disk
        with open(os.path.join(self.real_git_io_path, file_name), "r", encoding="utf-8") as f:
            file_data = f.read()
        return file_data


git_io = GitIO(real_git_url, GDE_PATH)

user_data = {
    "user1": {"password": "pass1",
              "security": None,
              "client_id": None},
    "user2": {"password": "pass2",
              "security": None,
              "client_id": None},

}


def get_user_by_client_id(client_id):
    for user in user_data:
        if user_data[user]["client_id"] == client_id:
            return user
    return None


def get_user_by_security(security):
    for user in user_data:
        if user_data[user]["security"] == security:
            return user
    return None


@PromptServer.instance.routes.post("/gde/logout")
async def logout(request):
    """
    This function is called when the user logs out
    """
    print("logout request")

    # get the user and pass from the json
    data = await request.json()
    client_id = data["client_id"]

    # find the user with the client id
    user = get_user_by_client_id(client_id)

    if user is None:
        # use json to send the data back
        return web.json_response({"error": "invalid client_id"}, status=401)

    # clear the security and client id
    user_data[user]["security"] = None
    user_data[user]["client_id"] = None

    # use json to send the data back
    return web.json_response({"success": True})


@PromptServer.instance.routes.post("/gde/login")
async def login(request):
    """
    This function is called when the attemnpts to log in:
    - validates login
    - sets up security
    - user and pass are in the json
    """
    print("login request")

    # get the user and pass from the json
    data = await request.json()
    try:
        client_id = data["client_id"]
    except:
        # this means the client id is not set yet so return logged_in False
        # use json to send the data back
        return web.json_response({"logged_in": False})
    try:
        user = data["username"]
        password = data["password"]
    except:
        # this means the client is just checking to see if it is already logged into the server
        # so check
        user = get_user_by_client_id(client_id)
        if user is None:
            # use json to send the data back
            print("user is none")
            return web.json_response({"logged_in": False})
        else:
            # use json to send the data back
            print("user is not none")
            return web.json_response({"logged_in": True})

    # validate the user and pass
    # for now the users and passes are right here in the code

    if user not in user_data:
        # use json to send the data back
        return web.json_response({"error": "invalid user"}, status=401)

    if password != user_data[user]["password"]:
        # use json to send the data back
        return web.json_response({"error": "invalid password"}, status=401)

    # if the user and pass are valid, then set up security
    # for now the security is just a random string that will be sent back to the user
    # in the future this will be a token that will be used to authenticate the user
    import uuid
    security = str(uuid.uuid4())

    # there is no middleware installed
    # so we will just store the security in memory
    # this is not secure, but it is good enough for now
    # store the security in memory
    user_data[user]["security"] = security
    user_data[user]["client_id"] = client_id

    # return json with the security
    return web.json_response({"security": security})


def hijack_prompt_server():
    """
    This function is called by the GDE when it starts up, it hijacks the prompt server
    for added security.
    """
    # remove the existing prompt route
    q = [i for i, s in enumerate(PromptServer.instance.routes) if ("/prompt" in str(s))]
    print(f"found {len(q)} routes to remove: {q}")
    for adj, i in enumerate(q):
        route = PromptServer.instance.routes._items.pop(i - adj)

        method = route.method
        path = route.path
        kwargs = route.kwargs

        if method == "GET":
            print(f"removing route {method} {path} {kwargs}")
            get_handler = route.handler

            @PromptServer.instance.routes.get(path)
            async def get_prompt(request):
                print("hijacked prompt get /prompt")
                print("headers:", request.headers)
                modified_func_response = await get_handler(request)
                return modified_func_response

        elif method == "POST":
            print(f"removing route {method} {path} {kwargs}")
            post_handler = route.handler

            @PromptServer.instance.routes.post(path)
            async def post_prompt(request: web.Request):
                # print("hijacked prompt post /prompt")
                # print("headers:", request.headers)

                json_data = await request.json()
                client_id = json_data.get("client_id", None)
                if client_id is None:
                    print("client_id is None")
                    return web.Response(text="client_id is None", status=400)
                else:
                    print("client_id:", client_id)
                    # find in users by client_id
                    try:
                        user = get_user_by_client_id(client_id)
                    except KeyError:
                        user = None

                    if user is None:
                        print("user not found")
                        return web.json_response({"error": "user not found"}, status=401)
                    else:
                        print(f"minimal security check passed for user {user}")

                modified_func_response: web.Response = await post_handler(request)

                print(modified_func_response.headers)

                return modified_func_response

                # return await post_handler(request)
                # return post_handler(request)


@PromptServer.instance.routes.post("/gde/git/user_graphs")
async def get_user_graphs(request):
    """
    Returns a list of available user graphs (dropdown options).
    """
    import os
    import subprocess
    data = await request.json()
    user = data.get("user", None)

    print("getting user graphs for user:", user)

    # Ensure that the git repo is in the correct state

    # Directory where the user graphs are stored
    user_graphs_dir = git_io.real_git_io_path

    # Ensure that the git repo is in the correct state
    git_io.ensure_git_state(user)

    # Check if the branch exists
    branch_exists = subprocess.call(["git", "rev-parse", "--verify", "--quiet", user])

    # If the branch does not exist, create it
    if branch_exists != 0:
        subprocess.call(["git", "checkout", "-b", user])

    # Pull the latest from the user's branch
    subprocess.call(["git", "pull"])

    # Get the list of user graph files
    user_graph_files = [f for f in os.listdir(user_graphs_dir) if os.path.isfile(os.path.join(user_graphs_dir, f))]

    # Convert the list of files to the desired format
    options_data = [{"value": f, "label": f} for f in user_graph_files if f != ""]

    return web.json_response(options_data)


@PromptServer.instance.routes.post("/gde/git/users")
async def get_users(request):
    """ the list of users is just the list of branches in the git repo"""
    import subprocess
    print("getting users")

    # Ensure that the git repo is in the correct state
    git_io.ensure_git_state("main")

    # Change to correct directory
    os.chdir(git_io.real_git_io_path)

    # Fetch updates from the remote
    subprocess.call(["git", "fetch", "--prune"])

    # List all remote branches
    remote_branches_raw = subprocess.check_output(["git", "branch", "-r"]).decode("utf-8").split("\n")
    remote_branches = [b.replace("origin/", "").strip() for b in remote_branches_raw]

    # List all local branches
    local_branches_raw = subprocess.check_output(["git", "branch"]).decode("utf-8").split("\n")
    local_branches = [b.replace("*", "").strip() for b in local_branches_raw]

    # Identify local branches not present in the remote branches
    branches_to_delete = [b for b in local_branches if b not in remote_branches]

    # Delete these local branches
    for branch in branches_to_delete:
        subprocess.call(["git", "branch", "-D", branch])

    # Now, get the updated list of local branches
    branches = subprocess.check_output(["git", "branch"]).decode("utf-8").split("\n")
    branches = [b.replace("*", "").strip() for b in branches]

    # Convert the list of branches to the desired format
    options_data = [{"value": b, "label": b} for b in branches if b != ""]

    return web.json_response(options_data)


@PromptServer.instance.routes.post("/gde/git/save")
async def save_to_git(request):
    """
    for git save version 1 each user will have their own branch,
    and the data will be saved to a file in the user's branch
    version > 1 may include other strategies
    """

    git_save_version = 1
    req_json = await request.json()
    print("saving to git")
    global fake_git
    req_json = await request.json()
    data = req_json.get("data", None)
    user_name = req_json.get("user_name", None)
    file_name = req_json.get("file_name", None)

    git_io.git_save_version_1(user_name, file_name, data)
    # Implement your actual Git saving logic here
    return web.Response(text="Saved to Git successfully.")


@PromptServer.instance.routes.post("/gde/git/load")
async def load_from_git(request):
    """use git_io to load the file from git"""
    print("loading from git")
    req_json = await request.json()
    user_name = req_json.get("user_name", None)
    file_name = req_json.get("file_name", None)

    # Implement your actual Git loading logic here
    file_data = git_io.git_load_version_1(user_name, file_name)
    return web.json_response({"data": file_data})


if __name__ != 'EternalKernelLiteGraphNodes.server_endpoints':
    hijack_prompt_server()
