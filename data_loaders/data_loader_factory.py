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
        data_path = config.train.valid_data_path if type == "validation" else config.inference.data_path
        if config.model.without_system_role:
            return InferenceDataLoaderWithoutSystem(data_path, config, tokenizer)
        else:
            return InferenceDataLoader(data_path, config, tokenizer)
