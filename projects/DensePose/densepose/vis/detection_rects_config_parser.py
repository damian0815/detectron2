from types import SimpleNamespace
import yaml

class DetectionRectsConfig:

    score_threshold = 0.5

    rect_definitions = [
    ]
    smpl_6980_vertex_groups = [

    ]

    def __init__(self, yaml_filename):
        with open(yaml_filename, "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.rect_definitions = data['rects']
                self.smpl_6980_vertex_groups = data['smpl_6980_vertex_groups']
                print(data)
            except yaml.YAMLError as exc:
                print(exc)


class DetectionRectsConfigParser:

    @classmethod
    def parse(cls, config_file_path):
        print("DetectionRectsConfigParser got config file path:", config_file_path)
        return DetectionRectsConfig(config_file_path)
