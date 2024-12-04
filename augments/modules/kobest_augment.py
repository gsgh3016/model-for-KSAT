import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANSWER, CHOICES, PARAGRAPH, QUESTION, VALID
from .langchain_manager import LangchainManager


class KoBESTAugment(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if data is None:
            # HuggingFace Hub에서 데이터 로드
            # https://huggingface.co/datasets/skt/kobest_v1
            dataset = load_dataset("skt/kobest_v1", "boolq")
            train_df = pd.DataFrame(dataset["train"])
            valid_df = pd.DataFrame(dataset["validation"])
            test_df = pd.DataFrame(dataset["test"])
            self.source_data = pd.concat([train_df, valid_df, test_df])
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")
        else:
            self.source_data = data

        # 출력이 문자열인 prompts/templates/question_choices_generation/question_choices_generation.txt 기반 체인 설정
        self.langchain_manager = LangchainManager(
            prompt_type="question_choices_generation",
            prompt_source="question_choices_generation.txt",
            output_type="json",
        )

        # 검증 프롬프트 설정
        self.validator = LangchainManager(
            prompt_type="question_choices_generation", prompt_source="validation.txt", output_type="bool"
        )

    def generate_problem(self, data: pd.Series):
        response = self.langchain_manager.invoke({PARAGRAPH: data[PARAGRAPH]})
        return response[QUESTION], response[CHOICES], response[ANSWER]

    def process(self):
        tqdm.pandas()
        self.source_data["result"] = self.source_data.progress_apply(self.generate_problem, axis=1)

        # 결과 분리
        self.source_data[QUESTION] = self.source_data["result"].apply(lambda x: x[0])
        self.source_data[CHOICES] = self.source_data["result"].apply(lambda x: x[1])
        self.source_data[ANSWER] = self.source_data["result"].apply(lambda x: x[2])

        # 임시 열 삭제
        self.source_data.drop(columns=["result", "label"], inplace=True)

        # 생성 데이터 검증
        self.source_data[VALID] = self.source_data.progress_apply(
            lambda row: self.validator.invoke(row.to_dict()), axis=1
        )

        # 필터링
        self.source_data = self.source_data[self.source_data[VALID]]
