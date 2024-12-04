from abc import ABC

import pandas as pd

from utils import DataVersionManager


class BaseProcessor(ABC):
    def __init__(self, data_path: str = "data/", experiment_data_path: str = "data/experiments/"):
        self.data_version_manager = DataVersionManager(data_path=data_path, experiment_data_path=experiment_data_path)
        self.source_data: pd.DataFrame = None
        self._created_columns: list[str] = []

    @property
    def created_columns(self) -> list[str]:
        return self._created_columns

    @created_columns.setter
    def created_columns(self, value: list[str]):
        self._created_columns = value

    def process(self) -> pd.DataFrame:
        pass

    def sample_data(self, n: int, seed: int = 1004):
        self.source_data = self.source_data.sample(n=n, random_state=seed)

    def inject_source_data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")
        self.source_data = data
