import pandas as pd
from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.inference.inference_data_loader import InferenceDataLoader


class InferenceDataLoaderWithoutSystem(InferenceDataLoader):
    def __init__(self, data_path: str, config: Config, tokenizer: PreTrainedTokenizerFast):
        super().__init__(data_path, config, tokenizer)

    def build_single_data(self, data: pd.Series, user_prompt: str):
        len_choices = len(data["choices"])

        result = {
            "id": data["id"],
            "messages": [
                {"role": "user", "content": "지문을 읽고 질문의 답을 구하세요.\n" + user_prompt},
            ],
            "len_choices": len_choices,
        }
        if "answer" in data:
            result["label"] = data["answer"]

        return result
