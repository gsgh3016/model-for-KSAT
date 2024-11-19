import re

from transformers import PreTrainedTokenizerFast
from trl import DataCollatorForCompletionOnlyLM

from models.special_tokens import SpecialTokens


def get_data_collator(tokenizer: PreTrainedTokenizerFast, response_template: str):
    """
    데이터를 모델에 입력하기 위해 처리하는 데이터 콜레이터를 반환합니다.

    응답 템플릿을 사용하여 모델이 생성해야 할 응답 부분만을 학습합니다.

    Args:
        tokenizer (PreTrainedTokenizerFast): 텍스트 데이터를 토큰화할 토크나이저.

    Returns:
        DataCollatorForLanguageModeling: 언어 모델링용 데이터 콜레이터.
    """
    return DataCollatorForCompletionOnlyLM(
        response_template=SpecialTokens.start_of_response,
        tokenizer=tokenizer,
    )


def extract_answer_from_response(response: str) -> str:
    """
    모델 응답에서 스페셜 토큰을 기반으로 정답을 추출하는 함수.

    Args:
        response (str): 모델이 생성한 텍스트.
        special_tokens (SpecialTokens): 스페셜 토큰 객체.

    Returns:
        str: 추출된 정답. 답변이 없으면 None 반환.
    """
    answer_pattern = re.compile(
        f"{re.escape(SpecialTokens.start_of_answer)}(.*?){re.escape(SpecialTokens.end_of_answer)}"
    )
    match = answer_pattern.search(response)
    if match:
        return match.group(1).strip()
    return None


def set_chat_template(tokenizer: PreTrainedTokenizerFast):
    """
    토크나이저에 채팅 템플릿을 설정하는 함수입니다.

    Args:
        tokenizer: 업데이트할 토크나이저.

    Returns:
        tokenizer: 업데이트된 토크나이저.
    """
    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        # chat_template 속성이 없거나 비어 있으면 기본 템플릿 설정
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
            "{% if system_message is defined %}{{ system_message }}{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        special_tokens_dict = {
            "additional_special_tokens": SpecialTokens.to_list() + ["<start_of_turn>", "<end_of_turn>"]
        }
    else:
        special_tokens_dict = {"additional_special_tokens": SpecialTokens.to_list()}

    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def prepare_tokenizer(tokenizer: PreTrainedTokenizerFast):
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
