import os
import re

import pandas as pd

from .prompt import load_template


def check_prompt_path(prompt_type: str, file_name: str):
    """
    프롬프트 경로를 검사하는 함수

    Args:
        prompt_type (str): 프롬프트 종류 - `prompts/templates/` 내에 있는 디렉토리 명
        file_name (str): 프롬프트 파일명 - `prompts./templates/{prompt_type}/ 내에 있는 txt를 포함한 파일명

    Raises:
        FileNotFoundError: 프롬프트 종류/파일명에 맞는 디렉토리/파일이 존재하지 않을 때
    """
    prompt_root_dir = os.path.join(__file__, "templates")
    prompt_type_dir = os.path.join(prompt_root_dir, prompt_type)
    prompt_file = os.path.join(prompt_type_dir, file_name)

    # prompt_type_dir: prompts/template/{prompt_type}
    if not os.path.exists(prompt_type_dir):
        raise FileNotFoundError(f"No {prompt_type} directory in {prompt_root_dir}")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"No {prompt_file} in {prompt_type_dir}")


def detect_input_variables(prompt_type: str, file_name: str) -> list[str]:
    """
    프롬프트에서 입력 변수를 감지하는 함수.
    중괄호로 감싸져 있는 변수 명을 읽어옵니다.

    NOTE: 영문자, 숫자, 언더바(_)외 중괄호 내부 문자는 감지하지 않습니다.

    ex) {"key": "value",}

    Args:
        prompt_type (str): 프롬프트 종류 - `prompts/templates/` 내에 있는 디렉토리 명
        file_name (str): 프롬프트 파일명 - `prompts./templates/{prompt_type}/ 내에 있는 txt를 포함한 파일명

    Returns:
        list[str]: 입력 변수 명
    """
    prompt = load_template(file_name=file_name, template_type=prompt_type)
    matches = re.findall(r"\{([a-zA-Z0-9_]+)}(?![^\{]*\})", prompt)
    return matches


def parse_input(data: pd.Series, prompt_type: str, file_name: str) -> dict[str, str]:
    pass
