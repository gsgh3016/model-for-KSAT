import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANALYSIS, CATEGORY
from .langchain_manager import LangchainManager


class Classification(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")

        # 출력이 문자열인 prompts/templates/data_classification/information_source_with_reasoning.txt 기반 체인 설정
        self.langchain_manager = LangchainManager(
            prompt_type="data_classification", prompt_source="information_source_with_reasoning.txt", output_type="json"
        )

    def process(self):
        tqdm.pandas()
        result = self.source_data.progress_apply(lambda row: self.langchain_manager.invoke(row), axis=1)
        result_df = result.apply(pd.Series)
        self.source_data[[ANALYSIS, CATEGORY]] = result_df
