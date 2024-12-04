import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser, OutputFixingParser
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from prompts import check_prompt_path, load_template, parse_input

STR = "str"
JSON = "json"
BOOL = "bool"


class LangchainManager:
    VALID_OUTPUT_TYPES = {STR, JSON, BOOL}

    def __init__(self, prompt_type: str, prompt_source: str, output_type: str):
        # 환경 변수 로드
        load_dotenv()

        # 프롬프트 경로 및 출력 유형 검증
        check_prompt_path(prompt_type=prompt_type, file_name=prompt_source)
        if output_type not in self.VALID_OUTPUT_TYPES:
            raise ValueError(f"Invalid output_type. Must be one of {self.VALID_OUTPUT_TYPES}")

        # 속성 초기화
        self.prompt_type = prompt_type
        self.prompt_source = prompt_source
        self.output_type = output_type
        self.llm = self._initialize_model("gpt-4o-mini", 0, None, None, 2)

        # 체인 설정
        self.chain = self._build_chain()

    @staticmethod
    def _initialize_model(model, temperature, max_tokens, timeout, max_retries) -> ChatOpenAI:
        """LLM 모델 초기화."""
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_chain(self) -> RunnableSerializable:
        """체인 빌드."""
        prompt_template = PromptTemplate.from_template(
            template=load_template(file_name=self.prompt_source, template_type=self.prompt_type)
        )

        # 출력 유형에 따른 체인 구성
        parser = self._get_output_parser()
        return prompt_template | self.llm | parser if parser else prompt_template | self.llm

    def _get_output_parser(self):
        """출력 유형에 따른 파서 반환."""
        if self.output_type == JSON:
            return OutputFixingParser.from_llm(JsonOutputParser(), llm=self.llm)
        elif self.output_type == BOOL:
            return OutputFixingParser.from_llm(BooleanOutputParser(), llm=self.llm)
        return None

    def invoke(self, data: pd.Series) -> bool | dict[str, str] | str:
        input_contents = parse_input(data=data, prompt_type=self.prompt_type, file_name=self.prompt_source)
        self.response: bool | dict[str, str] | BaseMessage = self.chain.invoke(input=input_contents)
        self._check_chain_output()
        return self.response.content if self.output_type == STR else self.response

    def _check_chain_output(self):
        if (
            (self.output_type == STR and not isinstance(self.response, BaseMessage))
            or (self.output_type == JSON and not isinstance(self.response, dict))
            or (self.output_type == BOOL and not isinstance(self.response, bool))
        ):
            raise TypeError(f"Wrong type: {type(self.response)} is invoked from chain")
