import os
from dataclasses import dataclass

import yaml


class Config:
    def __init__(self, config_file="config.yaml"):
        # 현재 파일(config.py)의 절대 경로를 구합니다.
        current_dir = os.path.dirname(__file__)
        # config.yaml의 절대 경로를 생성합니다.
        config_path = os.path.join(current_dir, config_file)
        # config.yaml 파일을 엽니다.
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

        self.model = ModelConfig(**self.raw_config["model"])
        self.common = CommonConfig(**self.raw_config["common"])
        self.peft = PeftConfig(**self.raw_config["peft"])
        self.sft = SftConfig(**self.raw_config["sft"])
        self.wandb = WandbConfig(**self.raw_config["wandb"])
        self.train = TrainConfig(**self.raw_config["train"])
        self.inference = InferenceConfig(**self.raw_config["inference"])


@dataclass
class ModelConfig:
    name_or_path: str
    response_template: str
    without_system_role: str


@dataclass
class CommonConfig:
    seed: int
    device: str


@dataclass
class PeftConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str
    task_type: str


@dataclass
class SftConfig:
    do_train: bool
    do_eval: bool
    lr_scheduler_type: str
    max_seq_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    logging_strategy: str
    logging_steps: int
    save_strategy: str
    eval_strategy: str
    save_total_limit: int
    save_only_model: bool
    report_to: str


@dataclass
class WandbConfig:
    project: str


@dataclass
class TrainConfig:
    data_path: str
    valid_data_path: str
    valid_output_path: str


@dataclass
class InferenceConfig:
    model_path: str
    data_path: str
    output_path: str
