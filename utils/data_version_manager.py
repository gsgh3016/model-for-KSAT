import re
from pathlib import Path

import pandas as pd
import yaml
from packaging.version import Version


class DataVersionManager:
    def __init__(self):
        # 프로젝트 디렉토리 및 설정 파일 경로 설정
        project_directory = Path.cwd()
        self.data_version_path = project_directory / "configs/data_version.yaml"

        # 설정 파일 로드
        with self.data_version_path.open("r", encoding="utf-8") as f:
            self.raw_yaml = yaml.safe_load(f)

        # 데이터 경로와 관련 버전 정보 설정
        self.data_path: Path = project_directory / self.raw_yaml["data_path"]
        self.experiment_data_path: Path = project_directory / self.raw_yaml["experiment_data_path"]
        self.latest_train_version = self.raw_yaml["latest_train_version"]
        self.latest_valid_version = self.raw_yaml["latest_valid_version"]
        self.latest_test_version = self.raw_yaml["latest_test_version"]
        self.latest_experiments_version = self.raw_yaml["latest_experiments_version"]
        self.experiments_integration = self.raw_yaml["experiments_integration"]

    def _find_matching_files(self, directory: Path, prefix: str) -> list[str]:
        """
        주어진 디렉토리에서 특정 접두사를 가진 파일 목록을 찾는 함수

        Args:
            directory (Path): 탐색할 디렉토리 경로
            prefix (str): 파일 이름의 접두사

        Returns:
            list[str]: 접두사와 일치하는 파일 이름 목록

        Raises:
            FileNotFoundError: 디렉토리가 존재하지 않는 경우
            NotADirectoryError: 경로가 디렉토리가 아닌 경우
        """
        pattern = re.compile(rf"^{prefix}_v\d+\.\d+\.\d+\.csv$")

        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path '{directory}' is not a directory.")

        # 정규표현식과 일치하는 파일 검색
        matching_files = [f.name for f in directory.iterdir() if f.is_file() and pattern.match(f.name)]
        if not matching_files:
            raise FileNotFoundError(f"No files matching prefix '{prefix}' found in '{directory}'.")

        return matching_files

    def _find_latest_file(self, directory: Path, prefix: str) -> str:
        """
        접두사와 일치하는 파일 중 가장 최신 파일을 반환

        Args:
            directory (Path): 탐색할 디렉토리 경로
            prefix (str): 파일 이름의 접두사

        Returns:
            str: 가장 최신 파일 경로
        """
        file_list = self._find_matching_files(directory=directory, prefix=prefix)
        file_list.sort()
        return directory / file_list[0]

    def search_latest_train_data(self) -> pd.DataFrame:
        """가장 최신의 학습 데이터를 로드"""
        latest_file = self._find_latest_file(self.data_path, "train")
        return pd.read_csv(latest_file)

    def search_latest_valid_data(self) -> pd.DataFrame:
        """가장 최신의 검증 데이터를 로드"""
        latest_file = self._find_latest_file(self.data_path, "valid")
        return pd.read_csv(latest_file)

    def search_latest_test_data(self) -> pd.DataFrame:
        """가장 최신의 테스트 데이터를 로드"""
        latest_file = self._find_latest_file(self.data_path, "test")
        return pd.read_csv(latest_file)

    def search_latest_experiments_data(self) -> dict[int, pd.DataFrame]:
        """
        실험 데이터에서 주요 버전별 최신 데이터를 로드

        Returns:
            dict[int, pd.DataFrame]: 주요 버전별 데이터프레임 딕셔너리, key: 버전 Major, value: 해당 Major 버전의 최신 데이터
        """
        paths = self._scan_experiments_paths()
        return {major: pd.read_csv(path) for major, path in paths.items()}

    def _scan_experiments_paths(self) -> dict[int, str]:
        """
        실험 데이터 디렉토리에서 시맨틱 버전(vX.Y.Z)을 가진 파일들을 검색하고 최신 버전을 반환
        """
        version_pattern = re.compile(r"v(\d+\.\d+\.\d+)")
        versions = {}

        # 실험 데이터 저장 경로 내 파일 명 저장
        files = [f for f in self.experiment_data_path.iterdir() if f.is_file()]

        # 시멘틱 버저닝으로 돼 있는 파일 명 추출
        for file in files:
            match = version_pattern.search(file.name)
            if match:
                version_str = match.group(1)
                version_obj = Version(version_str)

                major = version_obj.major
                if major not in versions or Version(versions[major][1]) < version_obj:
                    versions[major] = (file, version_str)

        return {major: info[0] for major, info in versions.items()}

    def search_experiments_integration_data(self) -> pd.DataFrame:
        """
        통합 실험 데이터를 로드
        """
        matching_files = [
            f for f in self.experiment_data_path.iterdir() if self.experiments_integration in f.name and f.is_file()
        ]

        if not matching_files:
            raise FileNotFoundError("No file with 'integration' in its name was found.")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files with 'integration' in their names were found: {matching_files}")

        return pd.read_csv(matching_files[0])
