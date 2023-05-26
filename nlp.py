
#import custom_nodes.Derfuu_ComfyUI_ModdedNodes.components.fields as field
import PIL.Image
import torch
import torchvision
import PIL.Image as Image

import doctest
# make sure to import ABC
from abc import ABC, abstractmethod

import os
import openai


class Config:
    api_key = os.environ.get('OPENAI_API_KEY')
    organization = os.environ.get('OPENAI_ORG')


NODE_CLASS_MAPPINGS = {}

openai.organization = Config.organization
openai.api_key = Config.api_key


class SimpleTextWidget:
    """base class for simple construction"""

    def __init__(self, func):
        self.func = func


    @classmethod
    def register(self, name):
        global NODE_CLASS_MAPPINGS

        # register the widget
        NODE_CLASS_MAPPINGS[name] = self

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "text": ("STRING", {"multiline": True}),
            },

        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "handler"

    def handler(self, text):
        ret = self.func(text)
        return (ret,)


try:
    import folder_paths
except:
    pass

import os
import numpy as np
import hashlib

### info for code completion AI ###
"""
all of these classes are plugins for comfyui and follow the same pattern
all of the images are torch tensors and it is unknown and unimportant if they are on the cpu or gpu
all image inputs are (B,W,H,C)

avoid numpy and PIL as much as possible
"""


class LLMCompletion(SimpleTextWidget):
    """uses the input text to call the specified LLM model and returns the output string"""

    def __init__(self):
        self.func = self.OAI_completion
        super().__init__(self.func)
        import uuid
        self.SSID = str(uuid.uuid4())

        from main import server_obj_holder
        self.server_string = server_obj_holder[0]["server_strings"]
        self.server_string[self.SSID] = "empty"

    def OAI_completion(self, text):
        out_message = [{"role": "user", "content": text}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=out_message
        )
        gpt_message = response["choices"][0]["message"]["content"].strip()
        self.server_string[self.SSID] = gpt_message
        return gpt_message


LLMCompletion.register("LLM_Completion")
