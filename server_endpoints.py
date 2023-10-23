import json
from server import PromptServer
from aiohttp import web
from importlib import reload
import logging
import os
from folder_paths import output_directory, input_directory, temp_directory

fake_git = None

@PromptServer.instance.routes.post("/gde/git/save")
async def save_to_git(request):
    print("saving to git")
    global fake_git
    fake_git = await request.json()
    print(fake_git)
    # Implement your actual Git saving logic here
    return web.Response(text="Saved to Git successfully.")

@PromptServer.instance.routes.post("/gde/git/load")
async def load_from_git(request):
    print("loading from git")
    global fake_git
    # Implement your actual Git loading logic here
    if fake_git is not None:
        return web.json_response(fake_git)
    else:
        return web.Response(text="No data found.", status=404)
