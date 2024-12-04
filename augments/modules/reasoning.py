from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import REASONING
from .langchain_manager import LangchainManager


class Reasoning(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/"):
        super().__init__(data_path, experiment_data_path)

        # data/train_v1.0.2.csv 로드 및 데이터 칼럼 세팅
        self.source_data = self.data_version_manager.get_latest_train_dataframe(major=1, minor=0, load_exp=False)
        self._created_columns += [REASONING]

        # 출력이 문자열인 prompts/templates/base/reasoning.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(prompt_type="base", prompt_source="reasoning.txt", output_type="str")

    def process(self):
        tqdm.pandas()
        self.source_data[REASONING] = self.source_data.progress_apply(
            lambda row: self.lanchain_manager.invoke(row), axis=1
        )
