import os
import shutil

import dotenv
import pandas as pd
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from configs import Config, create_peft_config, create_sft_config
from data_loaders import load_data, prepare_datasets, tokenize_dataset
from evaluation import compute_metrics, preprocess_logits_for_metrics
from models import load_model_and_tokenizer
from prompts import make_prompt
from utils import set_seed

dotenv.load_dotenv()

# 설정 로드
config = Config()

set_seed(config.training.seed)

df = load_data("data/train.csv")

# 모델, 토크나이저 로드 및 설정
model, tokenizer = load_model_and_tokenizer(config.model.name_or_path, config.training.device)


processed_dataset = []
for i, row in df.iterrows():
    user_message = make_prompt(row, template_type="base")

    processed_dataset.append(
        {
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{row['answer']}"},
            ],
            "label": row["answer"],
        }
    )
processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

# 토큰화
tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer)

# 데이터셋 분리
train_dataset, eval_dataset = prepare_datasets(
    tokenized_dataset,
    max_length=config.sft.max_seq_length,
    test_size=0.1,
    seed=config.training.seed,
)

response_template = "<start_of_turn>model"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# # wandb 초기화
# wandb.init(
#     project=config.wandb.project,
#     entity=config.wandb.entity,
#     name=config.wandb.name,
#     config=config.raw_config,  # 설정값을 wandb에 로깅
# )

peft_config = create_peft_config(config.peft)
sft_config = create_sft_config(config.sft)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
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
final_model_dir = "outputs/ko-gemma"
print(f"Saving final model to {config.sft.save_dir}...")
trainer.save_model(final_model_dir)
