import numpy as np
import torch
from transformers import PreTrainedTokenizerFast


def preprocess_logits_for_metrics(logits, labels, logit_idx: list[int]):
    """
    모델의 logits를 조정하여 정답 토큰 부분만 출력하도록 설정하는 함수.

    Args:
        logits: 모델의 출력 logits.
        labels: 실제 레이블 (사용하지 않음).
        tokenizer: 토크나이저 객체.

    Returns:
        조정된 logits 텐서.
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


def logits_to_predictions(logits):
    """
    logits를 확률로 변환하는 함수.

    Args:
        logits: 모델의 출력 logits(1, 2, 3, 4, 5에 대해서만 존재).
    """
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predicted_indices = np.argmax(probs, axis=-1).tolist()

    # 0~4 -> 1~5 변환 (문자열 형태)
    prediction_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    predictions = [prediction_map[idx] for idx in predicted_indices]
    return predictions


def preprocess_labels(labels, tokenizer: PreTrainedTokenizerFast):
    """
    레이블 데이터를 전처리하는 함수.
    """
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return [x.strip() for x in labels]
