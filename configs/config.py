import os

import yaml


class Config:
    def __init__(self, config_file="config.yaml"):
        # 현재 파일(config.py)의 절대 경로를 구합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # config.yaml의 절대 경로를 생성합니다.
        config_path = os.path.join(current_dir, config_file)
        # config.yaml 파일을 엽니다.
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def __getattr__(self, item):
        return self.config.get(item, None)

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)
