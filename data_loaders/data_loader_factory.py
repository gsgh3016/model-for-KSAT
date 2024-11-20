from transformers import PreTrainedTokenizerFast

from configs import Config
from data_loaders.inference.inference_data_loader import InferenceDataLoader
from data_loaders.inference.inference_data_loader_without_system import InferenceDataLoaderWithoutSystem
from data_loaders.train.train_data_loader import TrainDataLoader
from data_loaders.train.train_data_loader_without_system import TrainDataLoaderWithoutSystem


def build_data_loader(type: str, tokenizer: PreTrainedTokenizerFast, config: Config):
    if type == "train":
        if config.model.without_system_role:
            return TrainDataLoaderWithoutSystem(config, tokenizer)
        else:
            return TrainDataLoader(config, tokenizer)
    else:  # inference
        if config.model.without_system_role:
            return InferenceDataLoaderWithoutSystem(config.inference.data_path, tokenizer)
        else:
            return InferenceDataLoader(config.inference.data_path, tokenizer)
