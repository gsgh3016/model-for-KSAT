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
        self.training = TrainingConfig(**self.raw_config["training"])
        self.peft = PeftConfig(**self.raw_config["peft"])
        self.sft = SftConfig(**self.raw_config["sft"])
        self.wandb = WandbConfig(**self.raw_config["wandb"])


@dataclass
class ModelConfig:
    name_or_path: str


@dataclass
class TrainingConfig:
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
    save_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_strategy: str
    eval_strategy: str
    save_total_limit: int
    save_only_model: bool
    report_to: str


@dataclass
class WandbConfig:
    project: str
    entity: str
    name: str
