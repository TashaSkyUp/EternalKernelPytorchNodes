import os
import importlib
OLD_NODE_CLASS_MAPPINGS = {
    "Interpolation": "LatentInterpolation",
    "Image Pad To Match": "PadToMatch",
    "Image Stack": "StackImages",
    "Image B/C": "ImageBC",
    "Image RGBA Mod": "RGBA_MOD",
    "Image RGBA Lower Clip": "rgba_lower_clip",
    "Image RGBA Upper Clip": "rgba_upper_clip",
    "Image RGBA Merge": "rgba_merge",
    "Image RGBA Split": "rgba_split",
    "Select From Batch": "select_from_batch",
    "Quantize": "Quantize",
    "Select From RGB Similarity": "SelectFromRGBSimilarity",
    "Load Layered Image": "LoadImage",
    "Code Execution Widget": "ExecWidget",
    "Image Distance Mask": "ImageDistanceMask",
    "Image Text": "TextRender",
    "function": "FuncBase",
    "function render": "FuncRender",
    "function render image": "FuncRenderImage",
    "Tiny Txt 2 Img": "TinyTxtToImg",
    "Preview Image Test": "PreviewImageTest",
    "ETK_KSampler": "KSampler",
}
# reverse the dict
OLD_NODE_CLASS_MAPPINGS = {v: k for k, v in OLD_NODE_CLASS_MAPPINGS.items()}

NODE_CLASS_MAPPINGS = {}

# Try importing optional dependencies
try:
    import comfy.samplers
    use_generic_nodes = False
except ImportError:
    use_generic_nodes = True
    print("ETK> Failed to import comfy.samplers, assuming no comfyui skipping SD related nodes")

# Load classes from modules in the custom_nodes.EternalKernelLiteGraphNodes package
package_dir = os.path.dirname(os.path.abspath(__file__))
package_name = "custom_nodes.EternalKernelLiteGraphNodes"
for module_file in os.listdir(package_dir):
    if module_file.endswith('.py') and not module_file.startswith('__'):
        module_name = package_name + '.' + module_file[:-3]  # Removing .py extension
        try:
            # import directly from the file
            full_path = os.path.join(package_dir, module_file)
            with open(full_path, 'r') as f:
                code = f.read()
            exec(code, globals(), locals())
            for class_name in dir():
                class_obj = locals()[class_name]
                if isinstance(class_obj, type) and hasattr(class_obj, 'INPUT_TYPES'):
                    NODE_CLASS_MAPPINGS[class_name] = class_obj


            #module = importlib.import_module(module_name)
            #for class_name in dir(module):
            #    class_obj = getattr(module, class_name)
            #    if isinstance(class_obj, type) and hasattr(class_obj, 'INPUT_TYPES'):  # Checking if it's a class and has INPUT_TYPES attribute
            #        NODE_CLASS_MAPPINGS[class_name] = class_obj
        except ImportError:
            if not use_generic_nodes:
                print(f"Failed to import {module_name}, it may require comfy.samplers. Skipping...")


# use OLD_NODE_CLASS_MAPPINGS to add the old names to the new nodes
new_node_class_mappings = {}
for k, v in OLD_NODE_CLASS_MAPPINGS.items():
    if NODE_CLASS_MAPPINGS[k]:
        new_node_class_mappings[v] = NODE_CLASS_MAPPINGS[k]
        # mark it for removeal
        NODE_CLASS_MAPPINGS[k] = None
    else:
        print(f"ETK> Failed to find node {k} in new nodes, skipping...")
# remove the nodes with None
NODE_CLASS_MAPPINGS = {k: v for k, v in NODE_CLASS_MAPPINGS.items() if v is not None}
NODE_CLASS_MAPPINGS.update(new_node_class_mappings)

