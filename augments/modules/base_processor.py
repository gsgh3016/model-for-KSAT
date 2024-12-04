from abc import ABC, abstractmethod

import pandas as pd

from utils import DataVersionManager


class BaseProcessor(ABC):
    def __init__(
        self, data_path: str = "data/", experiment_data_path: str = "data/experiments/", data: pd.DataFrame = None
    ):
        self.data_version_manager = DataVersionManager(data_path=data_path, experiment_data_path=experiment_data_path)
        self.source_data: pd.DataFrame = data

    @property
    def data(self) -> pd.DataFrame:
        return self.source_data

    @abstractmethod
    def process(self) -> None:
        pass

    def sample_data(self, n: int, seed: int = 1004):
        """
        랜덤하게 샘플링해서 데이터를 저장하도록 하는 함수

        Args:
            n (int): 샘플링한 데이터 개수
            seed (int, optional): 설정할 시드 값. 기본은 1004.
        """
        self.source_data = self.source_data.sample(n=n, random_state=seed)
