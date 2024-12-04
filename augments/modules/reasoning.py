import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import REASONING
from .langchain_manager import LangchainManager


class Reasoning(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path, data)

        # data/train_v1.0.2.csv 로드 및 데이터 칼럼 세팅
        if data is None:
            self.source_data = self.data_version_manager.get_latest_train_dataframe(major=1, minor=0, load_exp=False)
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")
        else:
            self.source_data = data

        # 출력이 문자열인 prompts/templates/base/reasoning.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(prompt_type="base", prompt_source="reasoning.txt", output_type="str")

    def process(self):
        tqdm.pandas()
        self.source_data.drop(columns=["id"], inplace=True)
        self.source_data[REASONING] = self.source_data.progress_apply(
            lambda row: self.lanchain_manager.invoke(row), axis=1
        )
