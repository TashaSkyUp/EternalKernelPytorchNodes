import os
from custom_nodes.EternalKernelLiteGraphNodes.local_shared import ETK_PATH, GDE_PATH
def check_for_js_extension():
    # here we check to see that the js extension is installed at {GDE_PATH}/web/extensions/etk/ui_modifications.js
    # if it is not, we will copy it there from the ETK_PATH
    # check if the file exists

    import shutil
    for fn in ["ui_modifications.js", "ui_helpers.js"]:
        check_full_path = os.path.normpath(os.path.join(ETK_PATH, f"{GDE_PATH}/web/extensions/etk/{fn}"))
        etk_dir_file_mtime = os.path.getmtime(check_full_path)
        web_dir_file_mtime = os.path.getmtime(os.path.join(ETK_PATH, fn))

        # determine direction of copy
        if etk_dir_file_mtime > web_dir_file_mtime:
            # copy from etk to web
            copy_from = check_full_path
            copy_to = os.path.join(ETK_PATH, fn)
        else:
            # copy from web to etk
            copy_from = os.path.join(ETK_PATH, fn)
            copy_to = check_full_path

        # copy the file, give the user feedback, overwrite if necessary
        print(f"copying {copy_from} to {copy_to}, because the file is out of date")
        shutil.copyfile(copy_from, copy_to)
        print(f"copied {copy_from} to {copy_to}")


check_for_js_extension()