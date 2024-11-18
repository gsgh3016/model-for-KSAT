import evaluate
import numpy as np
import torch

# metric 로드
acc_metric = evaluate.load("accuracy")

# 정답 토큰 매핑
int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


# 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
def preprocess_logits_for_metrics(logits, labels, tokenizer):
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
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


# metric 계산 함수
def compute_metrics(evaluation_result, tokenizer):
    """
    평가 지표를 계산하는 함수.

    Args:
        evaluation_result: (logits, labels) 튜플.
        tokenizer: 토크나이저 객체.

    Returns:
        accuracy metric 딕셔너리.
    """
    logits, labels = evaluation_result

    # 토큰화된 레이블 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    # labels = list(map(lambda x: int_output_map[x], labels))
    labels = [x.split("<end_of_turn>")[0].strip() for x in labels]
    labels = [int_output_map[x] for x in labels]

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc
