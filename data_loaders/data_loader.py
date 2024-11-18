from ast import literal_eval

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from configs import Config
from prompts import make_prompt


class DataLoader:
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, config: Config):
        self.tokenizer = tokenizer

        df = pd.read_csv(data_path)
        df["choices"] = df["choices"].apply(literal_eval)
        df["question_plus"] = df["question_plus"].fillna("")

        dataset = self.preprocess_dataset(df)
        tokenized_dataset = self.tokenize_dataset(dataset)

        # 데이터셋 분리
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= config.sft.max_seq_length)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=config.training.seed)

        self.train_dataset = tokenized_dataset["train"]
        self.eval_dataset = tokenized_dataset["test"]

    def preprocess_dataset(self, df: pd.DataFrame) -> Dataset:
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
        return Dataset.from_list(processed_dataset)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        # 데이터 토큰화
        tokenized_dataset = dataset.map(
            self.tokenize,
            # remove_columns=list(processed_dataset.features), # 원본
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        return tokenized_dataset

    def tokenize(self, element):
        outputs = self.tokenizer(
            self.formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                self.tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts
