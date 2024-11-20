import pandas as pd
from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.train.train_data_loader import TrainDataLoader


class TrainDataLoaderWithoutSystem(TrainDataLoader):
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(config, tokenizer)

    def build_single_data(self, data: pd.Series, user_prompt: str):
        len_choices = len(data["choices"])

        return {
            "id": data["id"],
            "messages": [
                {"role": "user", "content": "지문을 읽고 질문의 답을 구하세요.\n" + user_prompt},
                {"role": "assistant", "content": f"{data['answer']}"},
            ],
            "label": data["answer"],
            "len_choices": len_choices,
        }
