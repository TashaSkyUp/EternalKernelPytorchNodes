import hashlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_list = {"list": ("LIST",)}
_func = {"func": ("FUNC", {"default": None})}
_func_extra = {"func_extra": ("FUNC",)}
_text = {"code": ("STRING", {"multiline": True, "default": "y mapped to output"})}
_image = {"image": ("IMAGE",)}
_string = {"string": ("STRING", {"default": None})}
_any = {"any": ("*", {"default": None})}

optional = lambda x: {"optional": {k: v for k, v in x.items()}}
required = lambda x: {"required": {k: v for k, v in x.items()}}
both = lambda a, b: {**a, **b}
many = lambda *dicts: {k: v for d in dicts for k, v in d.items()}  # non tested


def ETK_functional_base(cls):
    cls.FUNCTION = "func"
    cls.CATEGORY = "ETK/functional"
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    return cls


@ETK_functional_base
class FuncBase:
    """
    Base class for func nodes

    Allows for a function to be passed in as a callable
    and
    for a string to be passed in as code to be executed
    also
    returns the function as a callable

    """

    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        req = required(_text)
        opt = optional(
            both(
                both(
                    _func,
                    _func_extra
                ),
                _list
            )
        )
        use = both(req, opt)

        return use

    RETURN_TYPES = ("FUNC",)
    FUNCTION = "func"
    CATEGORY = "ETK/func"

    def func(self, **kwargs):
        import inspect
        self.globals = globals()
        self.locals = locals()
        self.locals["x1"] = kwargs.get("func", None)
        self.locals["x2"] = kwargs.get("func_extra", None)
        self.locals["x_list"] = kwargs.get("list", None)

        self.IN_FUNC = kwargs.get("func", None)
        self.EXTRA_IN_FUNC = kwargs.get("func_extra", None)
        self.exec_code = kwargs.get("code", None)
        self.ARGS = kwargs
        self.CODE = inspect.getsource(self.func)

        exec(self.exec_code, self.globals, self.locals)

        return (self.func,)


@ETK_functional_base
class FuncRender:
    """
    as input takes a function and a string of code
    optionally takes a list of arguments
    returns rendered results of types
    float, string, int, list
    """

    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        use = both(
            required(
                _text),
            optional(both(_list, _func)
                     )
        )
        return use

    RETURN_TYPES = ("FLOAT", "STRING", "INT", "LIST")
    FUNCTION = "func"
    CATEGORY = "ETK/func"

    def func(self, **kwargs):
        y_float = None
        y_string = None
        y_int = None
        func = kwargs.get("func", None)
        code = kwargs.get("code", None)

        globals = self.globals
        locals = self.locals

        locals["y_float"] = y_float
        locals["y_string"] = y_string
        locals["y_int"] = y_int
        locals["y_list"] = None

        locals["x"] = func
        locals["x_list"] = kwargs.get("list", None)
        # code should do something like:
        # func like : lambda x:["what"]*x
        # code like : 'f=[x for x in x(2)]'
        exec(code, globals, locals)
        self.globals = globals
        self.locals = locals

        return (locals["y_float"], locals["y_string"], locals["y_int"], locals["y_list"],)


@ETK_functional_base
class FuncRenderImage:
    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        use = both(
            required(
                both(_func, _text)),
            optional(_list)
        )
        return use

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    CATEGORY = "ETK/func"
    OUTPUT_NODE = True

    def func(self, func, code, **kwargs):
        globals = self.globals
        locals = self.locals
        locals["x"] = func
        locals["x_list"] = kwargs.get("list", None)
        # code should do something like:
        # func like : lambda x:["what"]*x
        # code like : 'f=[x for x in x(2)]'
        exec(code, globals, locals)
        self.globals = globals
        self.locals = locals

        return (locals["y"],)


@ETK_functional_base
class ExecWidget:
    import torch

    """runs eval on the given text"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional":
            {

                "code_str_in": ("STRING", {"multiline": True,
                                           "default": ""
                                           }),
                "image1_in": ("IMAGE",),
                "float1_in": ("FLOAT", {"multiline": False}),
                "string1_in": ("STRING", {"multiline": False}),
                "int1_in": ("INT", {"multiline": False}),

            },
            "required": {
                "text_to_eval": ("STRING",
                                 {"multiline": True,
                                  "default": "int_out=int_out\n"
                                             "float_out=float_out\n"
                                             "string_out=string_out\n"
                                             "image_out=image_out\n"
                                  }),
            }
        }

    CATEGORY = "text"
    RETURN_TYPES = ("STRING", "IMAGE", "FLOAT", "INT", "TUPLE", "CODE")
    FUNCTION = "pre_exec_handler"
    INTERNAL_STATE_DISPLAY_CODE = True
    OUTPUT_NODE = True

    def pre_exec_handler(self, **kwargs):
        image1_in = kwargs.get("image1_in", None)
        float1_in = kwargs.get("float1_in", 0.0)
        string1_in = kwargs.get("string1_in", "")
        int1_in = kwargs.get("int1_in", 0)
        tuple1_in = kwargs.get("tuple1_in", None)
        name = kwargs.get("name", "exec_func")
        text_to_eval = kwargs.get("text_to_eval", "")
        code_str_in = kwargs.get("code_str_in", "")
        if code_str_in != "":
            text_to_eval = code_str_in
        else:
            code_str_in = None

        ret = self.exec_handler(text_to_eval, image1_in, float1_in, string1_in, int1_in, tuple1_in, name)

        ret_formatted = {"ui": {"text": [ret[-1]]}, "result": ret}
        return ret_formatted

    def exec_handler(self, text_to_eval, image_obj: torch.Tensor = None, float_obj: float = 0.0, string_obj: str = "",
                     int_obj: int = 0, tuple_obj: tuple = None, name: str = "exec_func"):
        """
        >>> ExecWidget().exec_handler("2 + 3")
        '5'
        """

        new_locals = {"float_out": float_obj,
                      "image_out": image_obj,
                      "string_out": string_obj,
                      "int_out": int_obj,
                      "tuple_out": tuple_obj,
                      }

        try:
            if isinstance(text_to_eval, list):
                text_to_eval = text_to_eval[0]

            # evaluate for code tag
            if "<code>" in text_to_eval and "</code>" in text_to_eval:
                start = text_to_eval.find("<code>")
                end = text_to_eval.find("</code>")
                text_to_eval = text_to_eval[start + 6:end]
            # also evaluate for ```python tag

            if "```python" in text_to_eval and "```" in text_to_eval:
                start = text_to_eval.find("```python")
                end = text_to_eval.find("```", start + 8)
                text_to_eval = text_to_eval[start + 9:end]

            if "```" in text_to_eval:
                start = text_to_eval.find("```")
                end = text_to_eval.find("```", start + 2)
                text_to_eval = text_to_eval[start + 3:end]

            exec(text_to_eval, globals(), new_locals)
            source_code = text_to_eval

        except Exception as e:
            string_obj = str(e)
            source_code = text_to_eval

        string_obj = new_locals.get("string_out", string_obj)
        image_obj = new_locals.get("image_out", image_obj)
        float_obj = new_locals.get("float_out", float_obj)
        int_obj = new_locals.get("int_out", int_obj)
        tuple_obj = new_locals.get("tuple_out", tuple_obj)

        return (string_obj, image_obj, float_obj, int_obj, tuple_obj, text_to_eval)

    @classmethod
    def IS_CHANGED_NOPE(s, text_to_eval, image1_in: torch.Tensor = None, float1_in: float = 0.0, string1_in: str = "",
                        int1_in: int = 0, name: str = "exec_func"):

        m = hashlib.sha256()
        m.update(name.encode("utf-8"))
        return m.digest().hex()


@ETK_functional_base
class FuncListToList:
    """runs eval on the given text, with a list as input and output"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = required(
            both(_list,
                 _text)

        )
        opt = optional(_func)
        use = both(req, opt)
        return use

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("LIST",)
    FUNCTION = "func"

    def func(self, **kwargs):
        func = kwargs.get("func", None)
        code = kwargs.get("code", None)
        my_globals = globals()
        my_locals = locals()

        my_locals["x"] = func
        my_locals["x_list"] = kwargs.get("list", None)

        exec(code, my_globals, my_locals)

        return (my_locals["y_list"],)


@ETK_functional_base
class FuncStrToStr:
    """runs eval on the given text, with a string as input and outputs a string"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = required(both(_string, _text))
        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"

    def func(self, **kwargs):
        text = kwargs.get("string", None)
        x = text
        code = kwargs.get("code", None)
        my_globals = globals()
        my_locals = locals()

        exec(code, my_globals, my_locals)

        return (my_locals["y"],)


@ETK_functional_base
class FuncStrToList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = required(both(_string, _text))
        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("LIST",)
    FUNCTION = "func"

    def func(self, **kwargs):
        text = kwargs.get("string", None)
        code = kwargs.get("code", None)
        x = text
        my_globals = globals()
        my_locals = locals()

        exec(code, my_globals, my_locals)

        return (my_locals["y_list"],)


@ETK_functional_base
class FuncListToStr:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = required(both(_list, _text))
        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"

    def func(self, **kwargs):
        lst = kwargs.get("list", None)
        code = kwargs.get("code", None)
        x = lst
        x_list = lst
        my_globals = globals()
        my_locals = locals()

        exec(code, my_globals, my_locals)

        return (my_locals["y"],)


@ETK_functional_base
class GetFirstCodeBlock():
    """ returns the first code block in the text which is surrounded with ``` or
    ```python or
    <code> tags. If no code block is found, returns an empty string"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = required(_text)
        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"

    def func(self, **kwargs):
        text = kwargs.get("code", None)
        # find the code block
        if "```python" in text and "```" in text:
            start = text.find("```python")
            end = text.find("```", start + 8)
            text = text[start + 9:end]
        elif "```" in text:
            start = text.find("```")
            end = text.find("```", start + 2)
            text = text[start + 3:end]
        elif "<code>" in text and "</code>" in text:
            start = text.find("<code>")
            end = text.find("</code>")
            text = text[start + 6:end]
        else:
            text = ""
        return (text,)


@ETK_functional_base
class FuncAnysToImage:
    """runs eval on the given text, with 5 anys as input and outputs an image"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        req = optional(
            both(
                both({"any1": ("*", {"default": None})}, {"any2": ("*", {"default": None})}),
                both({"any3": ("*", {"default": None})}, {"any4": ("*", {"default": None})})
            ),
        )
        req["required"] = {}
        req["required"].update(_text)

        return req

    CATEGORY = "ETK/func"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"

    def func(self, **kwargs):
        code = kwargs.get("code", None)
        any1 = kwargs.get("any1", None)
        any2 = kwargs.get("any2", None)
        any3 = kwargs.get("any3", None)
        any4 = kwargs.get("any4", None)

        my_globals = globals()
        my_locals = locals()

        my_locals["any1"] = any1
        my_locals["any2"] = any2
        my_locals["any3"] = any3
        my_locals["any4"] = any4

        exec(code, my_globals, my_locals)

        return (my_locals["y"],)


if __name__ == "__main__":
    test = {}

    pass
