import pandas as pd
from transformers import PreTrainedTokenizerFast

from configs import Config

from .train_data_loader import TrainDataLoader


class TrainDataLoaderWithoutSystem(TrainDataLoader):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, config: Config):
        super().__init__(data_path, tokenizer, config)

    def build_single_data(self, data: pd.Series, user_prompt: str):
        return {
            "id": data["id"],
            "messages": [
                {"role": "user", "content": "지문을 읽고 질문의 답을 구하세요.\n" + user_prompt},
                {"role": "assistant", "content": f"{data['answer']}"},
            ],
            "label": data["answer"],
        }
