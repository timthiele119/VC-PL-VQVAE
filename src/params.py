import json
import os

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
GLOBAL_PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT, "config", "global_params.json")

class Params(dict):

    def __init__(self, config_file_path: str, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        with open(config_file_path) as f:
            content = json.load(f)
            for key, value in content.items():
                self[key] = value

    def __getattr__(self, item):
        return self.get(item)


global_params = Params(GLOBAL_PARAMS_FILE_PATH)
