from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.inference.inference_data_loader import InferenceDataLoader
from data_loaders.inference.inference_data_loader_without_system import InferenceDataLoaderWithoutSystem
from data_loaders.train.train_data_loader import TrainDataLoader
from data_loaders.train.train_data_loader_without_system import TrainDataLoaderWithoutSystem


def build_data_loader(type: str, data_path: str, tokenizer: PreTrainedTokenizerFast, config: Config):
    if type == "train":
        if config.model.without_system_role:
            return TrainDataLoaderWithoutSystem(data_path, tokenizer, config)
        else:
            return TrainDataLoader(data_path, tokenizer, config)
    else:  # inference
        if config.model.without_system_role:
            return InferenceDataLoaderWithoutSystem(data_path, tokenizer)
        else:
            return InferenceDataLoader(data_path, tokenizer)
