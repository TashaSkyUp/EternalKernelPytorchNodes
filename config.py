import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
config_file_path = os.path.join(this_file_path, "config.yaml")
# Load the YAML data when the module is imported
config_settings = read_yaml(config_file_path)
