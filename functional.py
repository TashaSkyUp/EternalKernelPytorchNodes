_func = {"func": ("FUNC",)}
_text = {"code": ("STRING", {"multiline": True, "default": "x=func(x)"})}

optional = lambda x: {"optional": {k: v for k, v in x.items()}}
required = lambda x: {"required": {k: v for k, v in x.items()}}
both = lambda a, b: {**a, **b}
many = lambda *dicts: {k: v for d in dicts for k, v in d.items()}  # non tested


f=lambda x : TinyTxtToImg.tinytxt2img(**kwa)
#TinyTxtToImg.tinytxt2img()


class FuncBase:
    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        use = both(optional(_func), required(_text))
        return use

    RETURN_TYPES = ("FUNC",)
    FUNCTION = "func"
    CATEGORY = "ETK/func"

    def func(self, code: str, func: callable = lambda x: x):

        globals = self.globals
        locals = self.locals
        locals["x"]=func
        # code should do something like:
        # func like : lambda x:["what"]*x
        # code like : 'f=[x for x in x(2)]'
        exec(code, globals, locals)

        return (locals["f"],)


class FuncRender:
    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        use = required(both(_func, _text))
        return use

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "func"
    CATEGORY = "ETK/func"

    def func(self, func: callable, code):

        globals = self.globals
        locals = self.locals
        locals["x"]=func
        # code should do something like:
        # func like : lambda x:["what"]*x
        # code like : 'f=[x for x in x(2)]'
        exec(code, globals, locals)
        self.globals = globals
        self.locals = locals

        return (locals["y"],)


class FuncRenderImage:
    def __init__(self):
        from custom_nodes.EternalKernelLiteGraphNodes.image import common_ksampler, TinyTxtToImg
        self.locals = locals()
        self.globals = globals()

    @classmethod
    def INPUT_TYPES(cls):
        use = required(both(_func, _text))
        return use

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    CATEGORY = "ETK/func"

    def func(self, func: callable, code):

        globals = self.globals
        locals = self.locals
        locals["x"]=func
        # code should do something like:
        # func like : lambda x:["what"]*x
        # code like : 'f=[x for x in x(2)]'
        exec(code, globals, locals)
        self.globals = globals
        self.locals = locals

        return (locals["y"],)

if __name__ == "__main__":
    test = {}

    pass
