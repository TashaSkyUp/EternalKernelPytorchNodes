import os
import shutil
import getpass
from custom_nodes.EternalKernelLiteGraphNodes.local_shared import ETK_PATH, GDE_PATH

import time


def set_file_mtime(file_path, mtime):
    """
    Set the modified time of the file to the specified time.

    :param file_path: Path of the file to modify.
    :param mtime: The new modified time as a timestamp (seconds since the epoch).
    """
    atime = os.path.getatime(file_path)  # keep the original access time
    os.utime(file_path, (atime, mtime))


def check_directory_permissions(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif not os.access(directory, os.R_OK | os.W_OK):
        raise PermissionError(f"Insufficient permissions for directory {directory}")


def check_for_js_extension():
    web_ext_path = os.path.normpath(os.path.join(GDE_PATH, "web", "extensions", "etk"))
    repo_file_path = os.path.join(ETK_PATH)

    # Check permissions for both directories
    check_directory_permissions(web_ext_path)
    check_directory_permissions(repo_file_path)

    for fn in ["ui_modifications.js", "ui_helpers.js"]:

        file_web_path = os.path.join(web_ext_path, fn)
        file_repo_path = os.path.join(repo_file_path, fn)

        try:
            file_web_date = os.path.getmtime(file_web_path)
        except FileNotFoundError:
            file_web_date = 0

        try:
            file_repo_date = os.path.getmtime(file_repo_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {file_repo_path} does not exist. Please check that the ETK_PATH is set correctly."
            )

        if file_web_date == file_repo_date:
            print(f"Files are identical: {file_web_path} = {file_repo_path}")
            continue

        if file_web_date > file_repo_date:
            direction = "web -> repo"
            from_file = file_web_path
            to_file = file_repo_path
            from_time = file_web_date
        else:
            direction = "repo -> web"
            from_file = file_repo_path
            to_file = file_web_path
            from_time = file_repo_date

        print(f"Copying {direction}: {from_file} -> {to_file}")
        shutil.copyfile(from_file, to_file)
        set_file_mtime(to_file, from_time)


# Print the current user
current_user = getpass.getuser()
print(f"This script is being executed as user: {current_user}")

check_for_js_extension()
