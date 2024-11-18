import os
import shutil

import dotenv
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from data_loaders import load_data, prepare_datasets, tokenize_dataset
from metrics import compute_metrics, preprocess_logits_for_metrics
from model_loader import add_special_tokens, load_model, load_tokenizer, prepare_tokenizer, set_chat_template
from prompts import make_prompt
from utils import set_seed

dotenv.load_dotenv()
set_seed(42)

df = load_data("data/train.csv")

# 모델 이름 또는 경로 지정
model_name_or_path = "beomi/gemma-ko-2b"

# 토크나이저 로드 및 설정
tokenizer = load_tokenizer(model_name_or_path)
tokenizer = set_chat_template(tokenizer)
tokenizer = add_special_tokens(tokenizer)
tokenizer = prepare_tokenizer(tokenizer)

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model_name_or_path, device=device)

# 토크나이저의 스페셜 토큰 수를 모델에 반영
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj"],
    bias="none",
    task_type="CAUSAL_LM",
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
train_dataset, eval_dataset = prepare_datasets(tokenized_dataset, max_length=1024, test_size=0.1, seed=42)

response_template = "<start_of_turn>model"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

sft_config = SFTConfig(
    do_train=True,
    do_eval=True,
    lr_scheduler_type="cosine",
    max_seq_length=1024,
    output_dir="outputs/checkpoint",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
    report_to="none",
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
print(f"Saving final model to {final_model_dir}...")
trainer.save_model(final_model_dir)
