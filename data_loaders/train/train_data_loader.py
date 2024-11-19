import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.base_data_loader import BaseDataLoader
from models import SpecialTokens


class TrainDataLoader(BaseDataLoader):
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(config.train.data_path, tokenizer)
        train_dataset = self.tokenize_dataset(self.dataset)
        train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= config.sft.max_seq_length)

        eval_df = self.read_csv(config.train.valid_data_path)
        self.origin_eval_dataset = self.preprocess_dataset(eval_df)

        eval_dataset = self.tokenize_dataset(self.origin_eval_dataset)
        eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= config.sft.max_seq_length)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def build_single_data(self, data: pd.Series, user_prompt: str):
        len_choices = len(data["choices"])

        response = (
            SpecialTokens.start_of_response
            # + SpecialTokens.start_of_reasoning  # Chain Of Thought Prompt 추가 시
            # + data["reasoning"]
            # + SpecialTokens.end_of_reasoning
            + SpecialTokens.start_of_answer
            + str(data["answer"])
            + SpecialTokens.end_of_answer
            + SpecialTokens.end_of_response
        )
        return {
            "id": data["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            # "label": data["answer"], # train, eval 둘다 안쓰임..
            "len_choices": len_choices,
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
