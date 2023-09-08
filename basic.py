class MultilineStringNode:
    """Noe that displays multiline text box"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "LIST")
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, Text):
        return (Text,)


class StrBreakout:
    """ node that breaks out a string into a:
    - Boolean
    - int
    - float
    - [string]
    - splitlines()
    - split(",")

    the first value is a description of the transformations tried
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": ("STRING", {
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "BOOL", "INT", "FLOAT", "LIST", "LIST", "LIST",)
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, Text):
        out_bool = False
        out_int = 0
        out_float = 0.0
        out_list = []
        out_lines = []
        out_comma = []
        notes = []
        try:
            out_bool = bool(Text)
            notes.append(f"bool worked")
        except Exception as e:
            notes.append(f"bool failed: {e}")

        try:
            out_int = int(Text)
            notes.append(f"int worked")
        except Exception as e:
            notes.append(f"int failed: {e}")
            pass
        try:
            out_float = float(Text)
            notes.append(f"float worked")
        except Exception as e:
            pass
        try:
            out_list = list(Text)
            notes.append(f"list worked")
        except Exception as e:
            notes.append(f"list failed: {e}")
            pass
        try:
            out_lines = Text.splitlines()
            notes.append(f"splitlines worked")
        except Exception as e:
            notes.append(f"splitlines failed: {e}")
            pass
        try:
            out_comma = Text.split(",")
            notes.append(f"split(',') worked")
        except Exception as e:
            notes.append(f"split(',') failed: {e}")
            pass
        return ("\n".join(notes), out_bool, out_int, out_float, out_list, out_lines, out_comma,)

class IntBreakout:
    """ node that breaks out an int into a:
    - Boolean
    - str(int)
    - float
    - [int]
    - list(range(int))
    - [None] * int

    the first value is a description of the transformations tried
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Int": ("INT", {
                    "default": 0,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "BOOL", "STRING", "FLOAT", "LIST", "LIST", "LIST",)
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, Int):
        out_bool = False
        out_str = ""
        out_float = 0.0
        out_list = []
        out_range = []
        out_none = []
        notes = []
        try:
            out_bool = bool(Int)
            notes.append(f"bool worked")
        except Exception as e:
            notes.append(f"bool failed: {e}")

        try:
            out_str = str(Int)
            notes.append(f"str worked")
        except Exception as e:
            notes.append(f"str failed: {e}")
            pass
        try:
            out_float = float(Int)
            notes.append(f"float worked")
        except Exception as e:
            pass
        try:
            out_list = list(Int)
            notes.append(f"list worked")
        except Exception as e:
            notes.append(f"list failed: {e}")
            pass
        try:
            out_range = list(range(Int))
            notes.append(f"range worked")
        except Exception as e:
            notes.append(f"range failed: {e}")
            pass
        try:
            out_none = [None] * Int
            notes.append(f"[None] * worked")
        except Exception as e:
            notes.append(f"[None] * failed: {e}")
            pass
        return ("\n".join(notes), out_bool, out_str, out_float, out_list, out_range, out_none,)



NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS["MultilineStringNode"] = MultilineStringNode
NODE_CLASS_MAPPINGS["StrBreakout"] = StrBreakout
NODE_CLASS_MAPPINGS["IntBreakout"] = IntBreakout
