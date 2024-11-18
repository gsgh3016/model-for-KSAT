from ast import literal_eval

import dotenv
import numpy as np
import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

from prompts import make_prompt
from utils import set_seed

if __name__ == "__main__":
    dotenv.load_dotenv()
    set_seed(42)

    model_path = "outputs/ko-gemma"

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load the test dataset
    # TODO Test Data 경로 입력
    test_df = pd.read_csv("data/test.csv")
    test_df["choices"] = test_df["choices"].apply(literal_eval)
    test_df["question_plus"] = test_df["question_plus"].fillna("")

    test_dataset = []
    for i, row in test_df.iterrows():
        user_message = make_prompt(row, template_type="base")
        len_choices = len(row["choices"])

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "len_choices": len_choices,
            }
        )
    infer_results = []

    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
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

    pd.DataFrame(infer_results).to_csv("data/output.csv", index=False)
