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


class ValueToAny:
    """takes INT,STRING,FLOAT,BOOL, and returns the value coded as *"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "INT": ("INT", {"default": None, }),
                "FLOAT": ("FLOAT", {"default": None, }),
                "STRING": ("STRING", {"default": None, }),
                "BOOL": ("BOOL", {"default": None, }),
            }
        }

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
            "optional": {
                "LATENT": ("LATENT", {"default": None, }),
                "CONDITIONING": ("CONDITIONING", {"default": None, }),
                "IMAGE": ("IMAGE", {"default": None, }),
                "MASK": ("MASK", {"default": None, }),
            }
        }

    RETURN_TYPES = ("*", "STRING",)
    RETURN_NAMES = ("*", "debug info",)
    FUNCTION = "get_value"
    CATEGORY = "conversion"

    def get_value(self, LATENT=None, CONDITIONING=None, IMAGE=None, MASK=None):
        if LATENT:
            dbg = f"found LATENT: {LATENT}"
            return (LATENT, dbg,)
        if CONDITIONING:
            dbg = f"found CONDITIONING: {CONDITIONING}"
            return (CONDITIONING, dbg,)
        if IMAGE:
            dbg = f"found IMAGE: {IMAGE}"
            return (IMAGE, dbg,)
        if MASK:
            dbg = f"found MASK: {MASK}"
            return (MASK, dbg,)
        dbg = f"found nothing, will return the first value that is not None; from LATENT,CONDITIONING,IMAGE,MASK"
        return (None, dbg)


NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS["MultilineStringNode"] = MultilineStringNode
NODE_CLASS_MAPPINGS["StrBreakout"] = StrBreakout
NODE_CLASS_MAPPINGS["IntBreakout"] = IntBreakout

NODE_CLASS_MAPPINGS["ValueToAny"] = ValueToAny
NODE_CLASS_MAPPINGS["IterableToAny"] = IterableToAny
NODE_CLASS_MAPPINGS["CallableToAny"] = CallableToAny
NODE_CLASS_MAPPINGS["ObjectToAny"] = ObjectToAny

NODE_CLASS_MAPPINGS["ETK_Tuple"] = TupleNode
