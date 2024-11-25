import os

import pandas as pd
import yaml


class DataVersionManager:
    def __init__(self):
        self.data_version_path = os.path.join(os.path.dirname(__file__), "../configs/data_version.yaml")
        with open(self.data_version_path, "r", encoding="utf-8") as f:
            self.raw_yaml = yaml.safe_load(f)
        self.data_path = self.raw_yaml["data_path"]
        self.experiment_data_path = self.raw_yaml["experiment_data_path"]
        self.latest_train_version = self.raw_yaml["latest_train_version"]
        self.latest_valid_version = self.raw_yaml["latest_valid_version"]
        self.latest_test_version = self.raw_yaml["latest_test_version"]
        self.latest_experiments_version = self.raw_yaml["latest_experiments_version"]
        self.experiments_integration = self.raw_yaml["experiments_integration"]

    def scan_data_files(self):
        self.train = self._scan_train_files()
        self.valid = self._scan_valid_files()
        self.test = self._scan_test_files()
        self.experiments_data = self._scan_experiments_files()
        self.integrated_data = self._scan_experiments_integration()

    def _scan_train_files(self) -> pd.DataFrame:
        pass

    def _scan_valid_files(self) -> pd.DataFrame:
        pass

    def _scan_test_files(self) -> pd.DataFrame:
        pass

    def _scan_experiments_files(self) -> list[pd.DataFrame]:
        pass

    def _scan_experiments_integration(self) -> pd.DataFrame:
        pass

    @property
    def train(self) -> pd.DataFrame:
        return self.train

    @property
    def valid(self) -> pd.DataFrame:
        return self.valid

    @property
    def test(self) -> pd.DataFrame:
        return self.test

    @property
    def experiments(self, i: int) -> pd.DataFrame:
        """
        실험 `i` 번의 최신 결과 데이터

        Args:
            i (int): `i`번 실험

        Returns:
            pd.DataFrame: 최신 `i`번 실험 데이터 결과
        """
        return self.experiments_data[i - 1]

    @property
    def integrated_data(self) -> pd.DataFrame:
        return self.integrated_data
