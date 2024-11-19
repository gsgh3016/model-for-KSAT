import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from configs import Config

from .data_loader import BaseDataLoader


class TrainDataLoader(BaseDataLoader):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, config: Config):
        super().__init__(data_path, tokenizer)

        tokenized_dataset = self.tokenize_dataset(self.dataset)

        # 데이터셋 분리
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= config.sft.max_seq_length)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=config.common.seed)

        self.train_dataset = tokenized_dataset["train"]
        self.eval_dataset = tokenized_dataset["test"]

    def build_single_data(self, data: pd.Series, user_prompt: str):
        return {
            "id": data["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": f"{data['answer']}"},
            ],
            "label": data["answer"],
        }

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
