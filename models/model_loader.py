import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from models.tokenizer_utils import add_special_tokens, prepare_tokenizer, set_chat_template


def load_model_and_tokenizer(model_name_or_path, device="cuda"):
    """
    모델과 토크나이저를 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.
        device (str): 모델을 로드할 디바이스 ('cuda' 또는 'cpu').

    Returns:
        model: 로드된 모델.
        tokenizer: 로드된 토크나이저.
    """
    model = load_model(model_name_or_path, device)
    tokenizer = load_tokenizer(model_name_or_path)

    # 토크나이저의 스페셜 토큰 수를 모델에 반영
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def load_model(model_name_or_path, device="cuda") -> PreTrainedModel:
    """
    모델을 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.
        device (str): 모델을 로드할 디바이스 ('cuda' 또는 'cpu').

    Returns:
        model: 로드된 모델.
    """
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.to(device)
    return model


def load_tokenizer(model_name_or_path):
    """
    토크나이저를 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.

    Returns:
        tokenizer: 로드된 토크나이저.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = set_chat_template(tokenizer)
    tokenizer = add_special_tokens(tokenizer)
    tokenizer = prepare_tokenizer(tokenizer)

    return tokenizer
