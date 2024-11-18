import os
import shutil

import dotenv
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from data_loaders import load_data
from utils import set_seed

dotenv.load_dotenv()
set_seed(42)

df = load_data("data/train.csv")

model = AutoModelForCausalLM.from_pretrained(
    "beomi/gemma-ko-2b",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "beomi/gemma-ko-2b",
    trust_remote_code=True,
)
tokenizer.chat_template = (
    "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}"
    "{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}"
    "{% set content = message['content'] %}{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
)

peft_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

dataset = Dataset.from_pandas(df)
PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

processed_dataset = []
for i in range(len(dataset)):
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

    # <보기>가 있을 때
    if dataset[i]["question_plus"]:
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=dataset[i]["paragraph"],
            question=dataset[i]["question"],
            question_plus=dataset[i]["question_plus"],
            choices=choices_string,
        )
    # <보기>가 없을 때
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=dataset[i]["paragraph"],
            question=dataset[i]["question"],
            choices=choices_string,
        )

    # chat message 형식으로 변환
    processed_dataset.append(
        {
            "id": dataset[i]["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{dataset[i]['answer']}"},
            ],
            "label": dataset[i]["answer"],
        }
    )

processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


def tokenize(element):
    outputs = tokenizer(
        formatting_prompts_func(element),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


# 데이터 토큰화
tokenized_dataset = processed_dataset.map(
    tokenize,
    remove_columns=list(processed_dataset.features),
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Tokenizing",
)
# 데이터 분리
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

response_template = "<start_of_turn>model"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)


# 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
def preprocess_logits_for_metrics(logits, labels):
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


# metric 로드
acc_metric = evaluate.load("accuracy")

# 정답 토큰 매핑
int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


# metric 계산 함수
def compute_metrics(evaluation_result):
    logits, labels = evaluation_result

    # 토큰화된 레이블 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


# pad token 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

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
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
