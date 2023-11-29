import os
from custom_nodes.EternalKernelLiteGraphNodes.local_shared import ETK_PATH, GDE_PATH


def check_for_js_extension():
    import shutil
    for fn in ["ui_modifications.js", "ui_helpers.js"]:
        etk_file_path = os.path.normpath(os.path.join(GDE_PATH, "web", "extensions", "etk", fn))
        web_file_path = os.path.join(ETK_PATH, fn)

        # Check if both files exist
        if os.path.exists(etk_file_path) and os.path.exists(web_file_path):
            # Check for read and write permissions
            if not os.access(etk_file_path, os.R_OK):
                raise PermissionError(f"Read permission denied for {etk_file_path}")
            if not os.access(web_file_path, os.W_OK):
                raise PermissionError(f"Write permission denied for {web_file_path}")

            etk_dir_file_mtime = os.path.getmtime(etk_file_path)
            web_dir_file_mtime = os.path.getmtime(web_file_path)

            # Determine direction of copy
            if etk_dir_file_mtime > web_dir_file_mtime:
                copy_from = etk_file_path
                copy_to = web_file_path
            else:
                copy_from = web_file_path
                copy_to = etk_file_path

            # Check if destination directory exists and has write permission
            dest_dir = os.path.dirname(copy_to)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            elif not os.access(dest_dir, os.W_OK):
                raise PermissionError(f"Write permission denied for directory {dest_dir}")

            # Copy the file and give feedback
            print(f"Copying {copy_from} to {copy_to}, because the file is out of date")
            shutil.copyfile(copy_from, copy_to)
            print(f"Copied {copy_from} to {copy_to}")
        else:
            print(f"One or both of the files {etk_file_path} and {web_file_path} do not exist.")


check_for_js_extension()
