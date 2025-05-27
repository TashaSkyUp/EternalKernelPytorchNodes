from .config import config_settings
from .utils import FileBackedDict
import json
TEST= False
try:
    from server import PromptServer
except ImportError:
    TEST=True
    # for now just exit

if not TEST:

    from aiohttp import web
    from importlib import reload
    import logging
    import os
    from folder_paths import output_directory, input_directory, temp_directory
    # from custom_nodes.EternalKernelLiteGraphNodes.local_shared import GDE_PATH, ETK_PATH
    from .local_shared import GDE_PATH, ETK_PATH
    from .config import config_settings

    fake_git = None


    # real_git_url = "https://github.com/story-squad/GDE_Graph_IO.git"  # ssh url
    # real_git_url = "https://github.com/Hopping-Mad-Games/COTRF.git"  # ssh url


    class GitIO:
        def __init__(self, i_real_git_url: str, i_gde_path: str):
            self.gde_path = i_gde_path
            self.real_git_io_path = self.setup_git_io(i_real_git_url)

        def setup_git_io(self, l_real_git_url: str):
            git_name = l_real_git_url.split("/")[-1].split(".")[0]

            real_git_path = os.path.join(self.gde_path, "data", "gitio", git_name)

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
            import subprocess
            """works in windows or linux, ensures that the git repo is in the correct state"""
            if not os.path.exists(self.real_git_io_path):
                raise Exception("git io path not found")
            if not os.path.exists(os.path.join(self.real_git_io_path, ".git")):
                raise Exception("git io error, not a git repo")

            def run_git_command(command: str):
                """Helper function to run a git command in the correct directory"""
                return subprocess.check_output(command, cwd=self.real_git_io_path, text=True).strip()

            # fetch first?
            run_git_command(["git", "fetch"])

            # get the current branch
            current_branch = run_git_command(["git", "branch", "--show-current"])

            ##################

            try:
                origin_branch_exists = not run_git_command(["git", "rev-parse", "--verify", "--quiet", "origin/" + user])
                origin_branch_exists = origin_branch_exists != ""
            except subprocess.CalledProcessError:
                origin_branch_exists = False

            try:
                local_branch_exists = run_git_command(["git", "rev-parse", "--verify", "--quiet", user])
                local_branch_exists = local_branch_exists != ""
            except subprocess.CalledProcessError:
                local_branch_exists = False

            if not origin_branch_exists and not local_branch_exists:
                print(f"git branch {user} does not exist locally or on the remote")
                # the branch does not exist locally or on the remote so create it here first
                # checkout new_user
                run_git_command(["git", "checkout", "new_user"])
                # create new branch
                run_git_command(["git", "branch", user])  # creates and checks out new branch
                # git checkout feature-branch
                run_git_command(["git", "checkout", user])
                # set the upstream to the remote
                # run_git_command(["git", "branch", "--set-upstream-to", "origin/" + user])
                # push the branch to the remote
                run_git_command(["git", "push", "-u", "origin", user])
                # fetch and pull it back down for good measure
                run_git_command(["git", "fetch"])
                run_git_command(["git", "pull"])

            elif not origin_branch_exists and local_branch_exists:
                print(f"git branch {user} exists locally but not on the remote")
                # the branch does not exist on the remote but does locally
                # set the upstream
                # run_git_command(["git", "branch", "--set-upstream-to", "origin/" + user])
                # push the branch to the remote
                run_git_command(["git", "push", "-u", "origin", user])
                # fetch and pull it back down for good measure
                run_git_command(["git", "fetch"])
                run_git_command(["git", "pull"])

            elif origin_branch_exists and not local_branch_exists:
                print(f"git branch {user} exists on the remote but not locally")
                # the branch does not exist locally but does on the remote
                # fetch and pull it down
                run_git_command(["git", "checkout", user])
                # pull
                run_git_command(["git", "pull"])

            elif origin_branch_exists and local_branch_exists:
                print(f"git branch {user} exists locally and on the remote")
                # the branch exists locally and on the remote
                # pull it down
                run_git_command(["git", "checkout", user])
                # set the upstream to the remote
                run_git_command(["git", "branch", "--set-upstream-to", "origin/" + user])

            current_branch = run_git_command(["git", "branch", "--show-current"])
            # if the current branch is not the user's branch, then switch to it
            if current_branch != user:
                print(f"git branch {user} is not the current branch, switching to it")
                # also check if the branch exists, if not then create it
                branches = run_git_command(["git", "branch"])
                if user not in branches:
                    run_git_command(["git", "branch", user])
                    # be verbose about what exactly is happening where
                    print(f"git branch {user} does not exist locally, creating it, and pushing it to the remote")
                    run_git_command(["git", "push", "--set-upstream", "origin", user])
                # switch to the user's branch
                run_git_command(["git", "checkout", user])

            # pull the latest from the user's branch
            run_git_command(["git", "pull"])

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


    git_io = None

    user_data_base = {
        "tasha_StSq": {
            "password": "[REDACTED]",
            "security": None,
            "client_id": None,
            "git_io": "StSq"
        },
        "tasha_COTRF": {
            "password": "[REDACTED]",
            "security": None,
            "client_id": None,
            "git_io": "COTRF"
        },
        "HMG": {
            "password": "[REDACTED]",
            "security": None,
            "client_id": None,
            "git_io": "HMG"
        },
        "API": {
            "password": "[REDACTED]",
            "security": None,
            "client_id": None,
            "git_io": "HMG"
        },
    }
    tmp_dir = config_settings["tmp_dir"]
    user_data_file = os.path.join(tmp_dir, 'user_data_tmp.json')
    user_data = FileBackedDict(user_data_file)

    if len(user_data.keys()) == 0:
        user_data.update(user_data_base)


    def get_user_by_client_id(client_id):
        for user in user_data:
            if user_data[user]["client_id"] == client_id:
                return user
        return None


    async def get_user_if_client_id_logged_in(client_id):
        try:
            user = get_user_by_client_id(client_id)
        except KeyError:
            user = None
        return user


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
        global git_io
        git_io = None

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
            has_client_id = True
        except:
            # this means the client id is not set yet so return logged_in False
            # use json to send the data back
            has_client_id = False
            return web.json_response({"logged_in": False})

        try:
            user = data["username"]
            password = data["password"]
            has_user_and_pass = True
        except:
            has_user_and_pass = False
            # this means the client is just checking to see if it is already logged into the server
            # so check
            user = get_user_by_client_id(client_id)
            if user is None:
                print("user is none")
                return web.json_response({"logged_in": False}, status=401)
            else:
                # send the client back its user so it can handle api calls correctly
                print("user recconected " + user)
                return web.json_response({"logged_in": True, "user": user}, status=200)

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

        global git_io

        # from . import config
        # config_settings = config.config_settings

        users_gitio = user_data[user]["git_io"]

        # users_gitio_path = os.path.abspath(os.path.join(
        #    GDE_PATH,
        #    config_settings["gitio"]["base"],
        #    users_gitio)
        # )

        try:
            users_gittio_url = config_settings["gitio"][users_gitio]
        except KeyError:
            return web.json_response({"error": "invalid gitio"}, status=401)

        git_io = GitIO(users_gittio_url, GDE_PATH)

        # return json with the security
        return web.json_response({"security": security})


    def hijack_prompt_server(route_path,
                             before_get_func: callable = None,
                             after_get_func: callable = None,
                             before_post_func: callable = None,
                             after_post_func: callable = None,
                             ):
        """
        This function is called by the GDE when it starts up, it hijacks the prompt server
        for added security.
        """
        from server import PromptServer

        # find the existing /prompt route
        rts = [i for i, s in enumerate(PromptServer.instance.routes) if (route_path == s.path)]
        print(f"found {len(rts)} routes to remove: {rts}")

        # remove the existing prompt route
        for i, rt_num in enumerate(rts):
            rt_to_pop = rt_num - i
            route = PromptServer.instance.routes._items.pop(rt_to_pop)

            # route = PromptServer.instance.routes._items[rt_to_pop]
            # myr = web.RouteTableDef()

            method = route.method
            path = route.path
            kwargs = route.kwargs

            if method == "GET":
                print(f"Hijacking route {method} {path} {kwargs}")
                get_handler = route.handler
                if not before_get_func:
                    if not after_get_func:
                        raise NotImplementedError

                if before_get_func and after_get_func:
                    raise NotImplementedError

                if before_get_func:
                    @PromptServer.instance.routes.get(path)
                    async def get_prompt(request):
                        part1 = await  before_get_func(request)
                        if isinstance(part1, web.Response):
                            return part1
                        if isinstance(part1, web.Request):
                            modified_func_response = await get_handler(part1)
                        return modified_func_response
                elif after_get_func:
                    @PromptServer.instance.routes.get(path)
                    async def get_prompt(request):
                        normal_handler_response = await get_handler(request)
                        modified_func_response = await  after_get_func(normal_handler_response)
                        return modified_func_response

            elif method == "POST":
                print(f"Hijacking route {method} {path} {kwargs}")
                post_handler = route.handler
                if not before_post_func:
                    if not after_post_func:
                        raise NotImplementedError

                if before_post_func and after_post_func:
                    raise NotImplementedError

                if before_post_func:
                    @PromptServer.instance.routes.post(path)
                    async def post_prompt(request):
                        part1 = await  before_post_func(request)
                        if isinstance(part1, web.Response):
                            return part1
                        if isinstance(part1, web.Request):
                            # modified_func_response = await post_handler(part1)
                            # if it has a json() method then it is a request, careful it might not have the attribute at all
                            if hasattr(part1, "json"):
                                modified_func_response = await post_handler(part1)
                        if isinstance(part1, dict):
                            a = await post_handler(part1)
                            b = part1
                            return a
                        return modified_func_response
                elif after_post_func:
                    @PromptServer.instance.routes.post(path)
                    async def post_prompt(request):
                        normal_handler_response = await post_handler(request)
                        modified_func_response = await after_post_func(normal_handler_response)
                        return modified_func_response


    from aiohttp import web


    @PromptServer.instance.routes.post("/gde/git/user_graphs")
    async def get_user_graphs(request):
        """
        Returns a list of available user graphs (dropdown options).
        """
        import os
        import subprocess

        if git_io == None:
            return web.json_response({"error": "git_io not set"}, status=401)

        def run_git_command(command: list, dir_path: str):
            """Helper function to run a git command in the specified directory"""
            return subprocess.call(command, cwd=dir_path)

        data = await request.json()
        client_id = data.get("client_id", None)
        user = data.get("user", None)
        if user not in user_data:
            user = get_user_by_client_id(client_id)

        if await get_user_if_client_id_logged_in(client_id) is None:
            return web.json_response({"error": "user not found"}, status=401)

        # user = data.get("user", None)
        print("getting user graphs for user:", user)

        # Directory where the user graphs are stored
        user_graphs_dir = git_io.real_git_io_path

        # Ensure that the git repo is in the correct state
        git_io.ensure_git_state(user)

        run_git_command(["git", "pull", "origin", user], user_graphs_dir)
        import os

        user_graph_files = [
            f for f in os.listdir(user_graphs_dir)
            if os.path.isfile(os.path.join(user_graphs_dir, f)) and f.endswith(".json")
        ]

        # Sort files by modification time, newest first
        user_graph_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(user_graphs_dir, x)),
            reverse=True
        )

        options_data = [{"value": f, "label": f} for f in user_graph_files if f != ""]
        # options_data = []

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


    # print(f"Hijacking route {method} {path} {kwargs}")
    # post_handler = route.handler
    #

    async def my_post_hijack_before(request: web.Request):
        json_data = await request.json()
        client_id = json_data.get("client_id", None)

        ret = request

        print("client_id:", client_id)
        if not isinstance(request, web.Request):
            print("request is not a web.Request")
            return request

        user = await get_user_if_client_id_logged_in(client_id)
        skp_lcl = config_settings["skip_login_for_local"]
        is_local_request = request.transport.get_extra_info('peername')[0] in (
            '127.0.0.1', '::1') if request.transport.get_extra_info('peername') else False

        if user:
            print("user:", user)
            ret = request
        elif skp_lcl and is_local_request:
            ret = request
        else:
            ret = web.json_response({"error": "user not found"}, status=401)

        return ret


    async def my_get_hijack_after(response):
        return response

    # ------------------------------------
    # WRAPPER: so we can override json()
    # ------------------------------------
    class ModifiedRequestWrapper:
        """
        A wrapper around the original `web.Request` that overrides
        the `json()` method to return our custom dictionary.
        """
        def __init__(self, original_request: web.Request, new_json: dict):
            self._original_request = original_request
            self._new_json = new_json

        def __getattr__(self, name):
            # Delegate everything except .json() to the original request
            return getattr(self._original_request, name)

        async def json(self):
            # Return our overridden data
            return self._new_json

    def fill_prompt(raw_json,**kwargs):
        for node_num,node in raw_json.items():
            inputs = node["inputs"]
            if node["class_type"]=="RequestInput":
                key = inputs["key"]
                if key in kwargs:
                    raw_json[node_num]["inputs"]["overridden_value"] = kwargs[key]
        return raw_json


    # ------------------------------------------------------------------
    # This function checks for "api_file" and if found, injects its JSON
    # into the request by returning a ModifiedRequestWrapper.
    # ------------------------------------------------------------------
    async def my_post_hijack_before_api_file(request: web.Request):
        import os
        import json

        # Convert the request body into a dict
        data = await request.json()
        api_file = data.get("api_file", None)

        if api_file:
            kwargs = data.get("kwargs", {})
            # Build path to the file in ./api_graphs
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_graphs_dir = os.path.join(current_dir, "api_graphs")
            file_path = os.path.join(api_graphs_dir, api_file)

            if not os.path.isfile(file_path):
                return web.json_response({"error": f"api_file '{api_file}' not found"}, status=400)

            with open(file_path, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)

            filled_json = fill_prompt(loaded_json,**kwargs)

            # Insert loaded_json into the "prompt" key
            data["prompt"] = filled_json

            # Return a new request-like object that yields `data` from .json()
            return ModifiedRequestWrapper(request, data)

        # If "api_file" isn't there, just return the original request
        return request

    if __name__ == 'EternalKernelLiteGraphNodes.server_endpoints':
        async def both_post_hijack_before(request: web.Request):
            thing1 = await my_post_hijack_before(request)
            thing2 = await my_post_hijack_before_api_file(thing1)
            return thing2

        hijack_prompt_server("/prompt", after_get_func=my_get_hijack_after, before_post_func=both_post_hijack_before)
        pass
