import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANALYSIS, CATEGORY
from .langchain_manager import LangchainManager


class Classification(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/"):
        super().__init__(data_path, experiment_data_path)

        # 추가할 데이터 칼럼 세팅
        self._created_columns += [ANALYSIS, CATEGORY]

        # 출력이 문자열인 prompts/templates/data_classification/information_source_with_reasoning.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(
            prompt_type="data_classification", prompt_source="information_source_with_reasoning.txt", output_type="json"
        )

    def process(self) -> pd.DataFrame:
        tqdm.pandas()
        self.source_data[[self._created_columns]] = self.source_data.progress_apply(
            lambda row: self.lanchain_manager.invoke(row), axis=1
        )
        return self.source_data
