import copy

from collections import OrderedDict


def etk_deep_copy(x, verbose=False):
    if type(x).__module__ == "numpy":
        return x.copy()
    elif type(x).__module__ == "torch":
        return x.clone()
    elif isinstance(x, list):
        return [etk_deep_copy(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(etk_deep_copy(item) for item in x)
    elif isinstance(x, dict):
        return {k: etk_deep_copy(v) for k, v in x.items()}
    elif isinstance(x, set):
        return set(etk_deep_copy(item) for item in x)
    elif isinstance(x, OrderedDict):
        return OrderedDict((k, etk_deep_copy(v)) for k, v in x.items())
    else:
        if verbose:
            print(f"copying {x}, of type {type(x)}")

        return copy.deepcopy(x)


import json
import os
from collections.abc import MutableMapping


class FileBackedDict(MutableMapping):
    FBD_MARKER = "__is_fbd__"  # Marker to indicate nested FileBackedDicts

    def __init__(self, file_path):
        # Ensure the provided path is absolute, raise an error if not
        if not os.path.isabs(file_path):
            raise ValueError(f"The path '{file_path}' is not an absolute path.")

        # Convert file_path to an absolute path (if needed)
        self.file_path = os.path.abspath(file_path)
        self._data = {}
        self._load()

    def _load(self):
        """Loads the data from the file into memory."""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                try:
                    raw_data = json.load(file)
                    # Convert any sub-dictionaries marked as FileBackedDicts
                    for key, value in raw_data.items():
                        if isinstance(value, dict) and value.get(self.FBD_MARKER, False):
                            # Create a nested FileBackedDict for each sub-dictionary
                            sub_dict_file = f"{self.file_path}_{key}.json"
                            self._data[key] = FileBackedDict(sub_dict_file)
                        else:
                            self._data[key] = value
                except json.JSONDecodeError:
                    self._data = {}
        else:
            self._data = {}

    def _save(self):
        """Saves the current data to the file."""
        # Save only the top-level keys that are not FileBackedDicts
        data_to_save = {}
        for key, value in self._data.items():
            if isinstance(value, FileBackedDict):
                # Mark this key as being backed by a FileBackedDict
                data_to_save[key] = {self.FBD_MARKER: True}
                value._save()  # Trigger save for the sub-dictionaries
            else:
                data_to_save[key] = value

        # Save the metadata about the keys
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, ensure_ascii=False, indent=4)

    def __getitem__(self, key):
        self._load() # Ensure the data is up-to-date
        return self._data[key]

    def __setitem__(self, key, value):
        self._load()  # Ensure the data is up-to-date
        if isinstance(value, dict):
            # Create a nested FileBackedDict for each sub-dictionary
            sub_dict_file = f"{self.file_path}_{key}.json"
            self._data[key] = FileBackedDict(sub_dict_file)
            self._data[key].update(value)
        else:
            self._data[key] = value
        self._save()

    def __delitem__(self, key):
        self._load()  # Ensure the data is up-to-date
        del self._data[key]
        self._save()

    def __iter__(self):
        self._load()  # Ensure the data is up-to-date
        return iter(self._data)

    def __len__(self):
        self._load()  # Ensure the data is up-to-date
        return len(self._data)

    def __repr__(self):
        self._load()  # Ensure the data is up-to-date
        return repr(self._data)

    def clear(self):
        """Clear the dictionary and save the changes to the file."""
        self._data.clear()
        self._save()

    def update(self, *args, **kwargs):
        self._load()  # Ensure the data is up-to-date
        """Update the dictionary with new data and save the changes to the file."""
        for key, value in dict(*args, **kwargs).items():
            self.__setitem__(key, value)
        self._save()