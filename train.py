import os
import shutil

import dotenv
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from configs import Config
from data_loaders import load_data, prepare_datasets, tokenize_dataset
from metrics import compute_metrics, preprocess_logits_for_metrics
from model_loader import add_special_tokens, load_model, load_tokenizer, prepare_tokenizer, set_chat_template
from prompts import make_prompt
from utils import set_seed

dotenv.load_dotenv()

# 설정 로드
config = Config()

set_seed(config.training.seed)

df = load_data("data/train.csv")

# 모델 이름 또는 경로 지정
model_name_or_path = config.model.name_or_path

# 토크나이저 로드 및 설정
tokenizer = load_tokenizer(model_name_or_path)
tokenizer = set_chat_template(tokenizer)
tokenizer = add_special_tokens(tokenizer)
tokenizer = prepare_tokenizer(tokenizer)

# 모델 로드
device = config.training.device
model = load_model(model_name_or_path, device=device)

# 토크나이저의 스페셜 토큰 수를 모델에 반영
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=config.peft.r,
    lora_alpha=config.peft.lora_alpha,
    lora_dropout=config.peft.lora_dropout,
    target_modules=config.peft.target_modules,
    bias=config.peft.bias,
    task_type=config.peft.task_type,
)

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


sft_config = SFTConfig(
    do_train=config.sft.do_train,
    do_eval=config.sft.do_eval,
    lr_scheduler_type=config.sft.lr_scheduler_type,
    max_seq_length=config.sft.max_seq_length,
    output_dir="outputs/checkpoint",
    per_device_train_batch_size=config.sft.per_device_eval_batch_size,
    per_device_eval_batch_size=config.sft.per_device_eval_batch_size,
    num_train_epochs=config.sft.num_train_epochs,
    learning_rate=config.sft.learning_rate,
    weight_decay=config.sft.weight_decay,
    logging_steps=config.sft.logging_steps,
    save_strategy=config.sft.save_strategy,
    eval_strategy=config.sft.eval_strategy,
    save_total_limit=config.sft.save_total_limit,
    save_only_model=config.sft.save_only_model,
    report_to=config.sft.report_to,
)

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
