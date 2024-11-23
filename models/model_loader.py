from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from configs import Config, create_bnb_config
from models.tokenizer_utils import prepare_tokenizer, set_chat_template
from utils import str_to_dtype


def load_model_and_tokenizer(model_name_or_path, config: Config):
    """
    모델과 토크나이저를 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.
        device (str): 모델을 로드할 디바이스 ('cuda' 또는 'cpu').

    Returns:
        model: 로드된 모델.
        tokenizer: 로드된 토크나이저.
    """
    model = load_model(model_name_or_path, config)
    tokenizer = load_tokenizer(model_name_or_path)
    return model, tokenizer


def load_model(model_name_or_path, config: Config) -> PreTrainedModel:
    """
    모델을 로드하는 함수입니다.

    Args:
        model_name_or_path (str): 사전 학습된 모델의 경로 또는 이름.
        device (str): 모델을 로드할 디바이스 ('cuda' 또는 'cpu').

    Returns:
        model: 로드된 모델.
    """
    # 양자화 설정 없을 시 None으로 설정(양자화가 없더라도 BitsAndBytesConfig가 들어가면 속도 저하됨)
    quantization_config = create_bnb_config(config.bnb) if config.bnb.load_in_4bit or config.bnb.load_in_8bit else None

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=str_to_dtype(config.model.torch_dtype),
        trust_remote_code=True,
    )

    # model.to(config.common.device)
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
    tokenizer = prepare_tokenizer(tokenizer)

    return tokenizer
