import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name_or_path, device="cuda"):
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
    return tokenizer


def set_chat_template(tokenizer):
    """
    토크나이저에 채팅 템플릿을 설정하는 함수입니다.

    Args:
        tokenizer: 업데이트할 토크나이저.

    Returns:
        tokenizer: 업데이트된 토크나이저.
    """
    tokenizer.chat_template = (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}"
        "{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}"
        "{% set content = message['content'] %}{% if message['role'] == 'user' %}"
        "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
        "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
    )
    return tokenizer


def add_special_tokens(tokenizer):
    """
    토크나이저에 스페셜 토큰을 추가하는 함수입니다.

    Args:
        tokenizer: 업데이트할 토크나이저.

    Returns:
        tokenizer: 업데이트된 토크나이저.
    """
    # special_tokens_dict = {
    #     "pad_token": tokenizer.eos_token,
    #     "additional_special_tokens": [
    #         "<start_of_turn>",
    #         "<end_of_turn>",
    #     ],
    # }
    # tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def prepare_tokenizer(tokenizer):
    """
    토크나이저를 설정하는 함수입니다.

    Args:
        tokenizer: 설정할 토크나이저.

    Returns:
        tokenizer: 설정된 토크나이저.
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer
