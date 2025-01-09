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


class TupleNode:
    """Node that displays a tuple"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Tuple": ("STRING", {
                    "default": "(None,None,)",
                }),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, Tuple):
        # evaluate the string safely step by step
        # type check
        if not isinstance(Tuple, str):
            raise TypeError("Tuple must be a string")

        # format check
        if not Tuple.startswith("(") or not Tuple.endswith(")"):
            raise ValueError("Tuple must start with '(' and end with ')'")

        # remove the brackets
        Tuple = Tuple.strip("()")

        # check for empty tuple
        if len(Tuple) == 0:
            return ((),)
        # check for commas
        if "," not in Tuple:
            raise ValueError("Tuple must have at least one comma")

        # split on commas
        Tuple = Tuple.split(",")

        # valid values are string,int,float,None, list,tuple,dict
        # construct the tuple
        ret = []
        for val in Tuple:
            val = val.strip()
            if val == "None":
                ret.append(None)
            elif val.startswith("[") and val.endswith("]"):
                # list
                val = val.strip("[]")
                val = val.split(",")
                ret.append(val)
            elif val.startswith("(") and val.endswith(")"):
                # tuple
                val = val.strip("()")
                val = val.split(",")
                ret.append(tuple(val))
            elif val.startswith("{") and val.endswith("}"):
                # dict
                val = val.strip("{}")
                val = val.split(",")
                ret.append(dict(val))
            elif val.startswith('"') and val.endswith('"'):
                # string
                ret.append(val.strip('"'))
            elif "." in val:
                # float
                ret.append(float(val))
            elif val == "":
                # empty
                pass
            else:
                # int
                ret.append(int(val))

        return (tuple(ret),)


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
                "list_dtype": (["str", "int", "float", ],),
            }
        }

    RETURN_TYPES = ("STRING", "BOOL", "INT", "FLOAT", "LIST", "LIST", "LIST",)
    RETURN_NAMES = ("Notes", "Bool", "Int", "Float", "List by chr", "list by lines", "list by commas",)
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, Text, list_dtype="str"):
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

        if list_dtype == "int":
            list_dtype = int
        elif list_dtype == "float":
            list_dtype = float
        else:
            list_dtype = str

        try:
            out_list = [list_dtype(val) for val in out_list]
            notes.append(f"list dtype worked")
        except Exception as e:
            notes.append(f"list dtype failed: {e}")
            pass
        try:
            out_lines = [list_dtype(val) for val in out_lines]
            notes.append(f"splitlines dtype worked")
        except Exception as e:
            notes.append(f"splitlines dtype failed: {e}")
            pass
        try:
            out_comma = [list_dtype(val) for val in out_comma]
            notes.append(f"split(',') dtype worked")
        except Exception as e:
            notes.append(f"split(',') dtype failed: {e}")
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


class SelectStrFromDict:
    """
    Selects a string from a dictionary
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict": ("DICT", {
                    "default": {},
                }),
                "key": ("STRING", {
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = "text"

    def get_value(self, dict, key):
        if key in dict:
            return (dict[key],)
        return ("",)


class StringToAny:
    """takes STRING, and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, string):
        try:
            ret_str = str(string)
            dbg = f"found STRING: {string}"
        except Exception as e:
            ret_str = None
            dbg = f"failed to verify STRING: {string} error: {e}"

        return (ret_str, dbg,)


class ValueToAny:
    """takes INT,STRING,FLOAT,BOOL, and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        ret = {
            "required": {},
            "optional": {
                "INT": ("INT", {"forceInput": True}),
                "FLOAT": ("FLOAT", {"forceInput": True}),
                "STRING": ("STRING", {"forceInput": True}),
                "BOOL": ("BOOL", {"forceInput": True}),
            }}
        return ret

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, INT=None, FLOAT=None, STRING=None, BOOL=None):
        if INT:
            dbg = f"found INT: {INT}"
            return (INT, dbg,)
        if FLOAT:
            dbg = f"found FLOAT: {FLOAT}"
            return (FLOAT, dbg,)
        if STRING:
            dbg = f"found STRING: {STRING}"
            return (STRING, dbg,)
        if BOOL:
            dbg = f"found BOOL: {BOOL}"
            return (BOOL, dbg,)
        dbg = f"found nothing, will return the first value that is not None; from INT,FLOAT,STRING,BOOL"
        return (None, dbg)


class IterableToAny:
    """takes LIST,DICT, and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "LIST": ("LIST", {"default": None, }),
                "DICT": ("DICT", {"default": None, }),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, LIST=None, DICT=None):
        if not isinstance(LIST, list):
            return (LIST, f"LIST is not a list: {LIST} is type: {type(LIST)}")
        if LIST:
            dbg = f"found LIST: {LIST}"
            return (LIST, dbg,)
        if DICT:
            dbg = f"found DICT: {DICT}"
            return (DICT, dbg,)
        dbg = f"found nothing, will return the first value that is not None; from LIST,DICT"
        return (None, dbg)


class CallableToAny:
    """takes FUNC, (others soon) and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "FUNC": ("FUNC", {"default": None, }),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, FUNC=None):
        if FUNC:
            dbg = f"found FUNC: {FUNC}"
            return (FUNC, dbg,)
        dbg = f"found nothing, will return the first value that is not None; from FUNC"
        return (None, dbg)


class ObjectToAny:
    """ takes non-primitive objects like TENSORS and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "0", })},
            "optional": {
                "LATENT": ("LATENT", {"default": None, }),
                "CONDITIONING": ("CONDITIONING", {"default": None, }),
                "IMAGE": ("IMAGE", {"default": None, }),
                "MASK": ("MASK", {"default": None, }),
                "CLIP": ("CLIP", {"default": None, }),
                "MODEL": ("MODEL", {"default": None, }),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, label,
                  LATENT=None,
                  CONDITIONING=None,
                  IMAGE=None,
                  MASK=None,
                  CLIP=None,
                  MODEL=None,
                  ):
        if LATENT:
            dbg = f"found LATENT: {LATENT}"
            return (LATENT, dbg,)

        if CONDITIONING:
            dbg = f"found CONDITIONING: {CONDITIONING}"
            return (CONDITIONING, dbg,)

        if IMAGE is not None:
            dbg = f"found IMAGE: {IMAGE}"
            return (IMAGE, dbg,)

        if MASK:
            dbg = f"found MASK: {MASK}"
            return (MASK, dbg,)

        if MODEL:
            dbg = f"found MODEL: {MODEL}"
            return (MODEL, dbg,)
        if CLIP:
            dbg = f"found CLIP: {CLIP}"
            return (CLIP, dbg,)

        dbg = f"found nothing, will return the first value that is not None; from LATENT,CONDITIONING,IMAGE,MASK"
        return (None, dbg)


from torch import Tensor
from comfy.model_patcher import ModelPatcher

COMFY_PYTHON_CLAS_NAMES = {
    "INT": int,
    "FLOAT": float,
    "STRING": str,
    "BOOLEAN": bool,
    "TUPLE": tuple,
    "LIST": list,
    "DICT": dict,
    "FUNC": callable,
    "LATENT": dict,
    "CONDITIONING": dict,
    "IMAGE": Tensor,
    "MASK": Tensor,
    "LLLM_MESSAGES": list,
    "LITELLM_MODEL": str,
    "MODEL": ModelPatcher,
}


def value_try(val, cls):
    try:
        return cls(val)
    except:
        return None


class AnyToValue:
    """
    takes * and returns it in the node systems common value types
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("*", {
                    "default": None,
                }),
            }}

    RETURN_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "TUPLE",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, A):
        int_o = value_try(A, int)
        float_o = value_try(A, float)
        str_o = value_try(A, str)
        bool_o = value_try(A, bool)
        tuple_o = value_try(A, tuple)

        return (int_o, float_o, str_o, bool_o, tuple_o,)


class AnyToIterable:
    """
    takes * and returns it in the node systems common value types
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Any": ("*", {
                    "default": None,
                }),
            }}

    RETURN_TYPES = ("LIST", "DICT",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, Any):
        list_o = None
        dict_o = None

        if isinstance(Any, list):
            list_o = Any

        if isinstance(Any, dict):
            dict_o = Any

        return (list_o, dict_o,)


class AnyToObject:
    """
    like AnyToIterable but for Image,latent,conditioning,model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Any": ("*", {
                    "default": None,
                }),
            }}

    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "MODEL")
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, x):
        image_o = x
        latent_o = x
        conditioning_o = x
        model_o = x

        return (image_o, latent_o, conditioning_o, model_o,)


class TimeThis:
    """
    Pass through node that records the current time to a given variable name
    then when called again reports the time difference
    """

    start_times = {}
    end_times = {}
    last_activator = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "var_name": ("STRING", {"default": "timer0", }),
                "pass_through": ("*", {"default": None, }),
                "start": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "activator": ("*")
            }

        }

    INTERNAL_STATE_DISPLAY = "text_display"
    RETURN_TYPES = ("STRING", "FLOAT", "OBJECT")
    RETURN_NAMES = ("Notes", "Time", "Object")
    FUNCTION = "time_this"
    CATEGORY = "utils"

    def time_this(self, var_name, pass_through, start, activator=None):
        import time

        if start:
            self.start_times[var_name] = time.time()
            print(f"Timer started for {var_name}")
            ret = ("Timer started", 0.0, pass_through)
        else:
            if var_name not in self.start_times:
                return ("Timer not started", 0.0, pass_through)
            self.end_times[var_name] = time.time()
            time_diff = self.end_times[var_name] - self.start_times[var_name]
            print(f"Timer ended for {var_name} time taken: {time_diff}")
            ret = (f"Time taken: {time_diff}", time_diff, pass_through)

        ret = {"ui": {"text": (ret[0],)}, "result": ret}
        return ret


class Text_ETK:
    """
    to provide text to the model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "text"

    def fun(self, text):
        return (text,)


class StringsToList:
    """
    take two strings and return them as a list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("LIST", {"default": [], "forceInput": True}),
                "string1": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "string2": ("STRING", {"default": "", "forceInput": True}),
            }

        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "fun"
    CATEGORY = "text"

    def fun(self, list, string1, string2=""):
        return (list + [string1, string2],)


# from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS["Text_ETK"] = Text_ETK
NODE_CLASS_MAPPINGS["Text _O"] = Text_ETK
NODE_CLASS_MAPPINGS["MultilineStringNode"] = MultilineStringNode
NODE_CLASS_MAPPINGS["StrBreakout"] = StrBreakout
NODE_CLASS_MAPPINGS["IntBreakout"] = IntBreakout

NODE_CLASS_MAPPINGS["ValueToAny"] = ValueToAny
NODE_CLASS_MAPPINGS["IterableToAny"] = IterableToAny
NODE_CLASS_MAPPINGS["CallableToAny"] = CallableToAny
NODE_CLASS_MAPPINGS["ObjectToAny"] = ObjectToAny

NODE_CLASS_MAPPINGS["ETK_Tuple"] = TupleNode
NODE_CLASS_MAPPINGS["AnyToValue"] = AnyToValue
NODE_CLASS_MAPPINGS["AnyToIterable"] = AnyToIterable
NODE_CLASS_MAPPINGS["TimeThis"] = TimeThis
NODE_CLASS_MAPPINGS["AnyToObject"] = AnyToObject

NODE_CLASS_MAPPINGS["SelectStrFromDict"] = SelectStrFromDict

NODE_CLASS_MAPPINGS["StringToAny"] = StringToAny
NODE_CLASS_MAPPINGS["StringsToList"] = StringsToList
