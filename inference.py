import dotenv
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutput

from data_loaders import InferenceDataLoader
from utils import set_seed


def inference(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, dataset: Dataset) -> pd.DataFrame:
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs: CausalLMOutput = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            probs = (
                torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    return pd.DataFrame(infer_results)


if __name__ == "__main__":
    dotenv.load_dotenv()
    set_seed(42)

    model_path = "outputs/ko-gemma"
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_loader = InferenceDataLoader("data/test.csv", tokenizer)

    infer_results = inference(model, tokenizer, data_loader.dataset)

    infer_results.to_csv("data/output.csv", index=False)
