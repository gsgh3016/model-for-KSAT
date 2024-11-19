import argparse
import os
import shutil

import dotenv
from trl import SFTTrainer

from configs import Config, create_peft_config, create_sft_config
from data_loaders import build_data_loader
from evaluation import compute_metrics, preprocess_logits_for_metrics
from models import get_data_collator, load_model_and_tokenizer
from utils import set_seed


def train(config: Config):
    # # wandb 초기화
    # wandb.init(
    #     project=config.wandb.project,
    #     entity=config.wandb.entity,
    #     name=config.wandb.name,
    #     config=config.raw_config,  # 설정값을 wandb에 로깅
    # )

    model, tokenizer = load_model_and_tokenizer(config.model.name_or_path, config.common.device)

    data_loader = build_data_loader("train", config.train.data_path, tokenizer, config)

    peft_config = create_peft_config(config.peft)
    sft_config = create_sft_config(config.sft)

    trainer = SFTTrainer(
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        data_collator=get_data_collator(tokenizer, config.model.response_template),
        tokenizer=tokenizer,
        compute_metrics=lambda eval_res: compute_metrics(eval_res, tokenizer),
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, tokenizer),
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()

    # 학습 종료 후 체크포인트 삭제
    if os.path.exists(sft_config.output_dir):
        print(f"Deleting checkpoints in {sft_config.output_dir}...")
        shutil.rmtree(sft_config.output_dir)  # 체크포인트 디렉토리 삭제

    # 최종 모델 저장
    print(f"Saving final model to {config.train.save_path}...")
    trainer.save_model(config.train.save_path)


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    train(config)
