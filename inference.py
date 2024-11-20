import argparse

import dotenv
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from configs import Config
from data_loaders import build_data_loader
from models import predict
from utils import set_seed


def inference(config: Config):
    model = AutoPeftModelForCausalLM.from_pretrained(config.inference.model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.inference.model_path)

    data_loader = build_data_loader("inference", tokenizer, config)

    prediction = predict(model, tokenizer, data_loader.dataset)

    prediction.to_csv(config.inference.output_path, index=False)


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    inference(config)
