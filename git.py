#class PromptTemplate:
#    """replaces the text in the given text string with a given other text at some key positions"""
#
#    @classmethod
#    def INPUT_TYPES(s):
#        return {"required":
#            {
#                "text": ("STRING", {"multiline": True}),
#                "replacement": ("STRING", {"multiline": True}),
#                 "key1": ("STRING", {"multiline": False}),
#             }
#         }
#
#     CATEGORY = "text"
#     RETURN_TYPES = ("STRING",)
#     FUNCTION = "prompt_template_handler"
#
#     def prompt_template_handler(self, text: str, replacement, key1):
#         """
#         >>> PromptTemplate().prompt_template_handler("hello world", "universe", "world")
#         'hello universe'
#         """
#         return (text.replace(key1, replacement),)