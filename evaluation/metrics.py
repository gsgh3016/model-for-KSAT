import evaluate
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

# metric 로드
acc_metric = evaluate.load("accuracy")


def compute_metrics(evaluation_result, tokenizer: PreTrainedTokenizerFast):
    """
    평가 지표를 계산하는 함수.

    Args:
        evaluation_result: (logits, labels) 튜플.
        tokenizer: 토크나이저 객체.

    Returns:
        accuracy metric 딕셔너리.
    """
    logits, labels = evaluation_result

    labels = preprocess_labels(labels, tokenizer)

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


def preprocess_labels(labels, tokenizer: PreTrainedTokenizerFast):
    """
    레이블 데이터를 전처리하는 함수.
    """
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = [x.split("<end_of_turn>")[0].strip() for x in labels]
    return [int_output_map[x] for x in labels]
