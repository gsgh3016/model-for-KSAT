import re
from pathlib import Path

import pandas as pd
import yaml
from packaging.version import Version


class DataVersionManager:
    """
    데이터 버전 관리를 담당하는 클래스.

    - 학습, 검증, 추론, 실험 데이터의 최신 버전을 탐색하고 로드.
    - `data_version.yaml` 파일을 업데이트하여 최신 버전을 반영.
    """

    def __init__(self):
        # 프로젝트 디렉토리 및 설정 파일 경로 설정
        project_directory = Path.cwd()
        self.data_version_path = project_directory / "configs/data_version.yaml"

        # 설정 파일 로드
        with self.data_version_path.open("r", encoding="utf-8") as f:
            self.raw_yaml = yaml.safe_load(f)

        # 데이터 경로와 최신 버전 정보 초기화
        self.data_path: Path = project_directory / self.raw_yaml["data_path"]
        self.experiment_data_path: Path = project_directory / self.raw_yaml["experiment_data_path"]
        self.latest_version = {
            "train": self.raw_yaml["latest_train_version"],
            "valid": self.raw_yaml["latest_valid_version"],
            "test": self.raw_yaml["latest_test_version"],
            "exp": self.raw_yaml["latest_experiments_version"],
        }
        self.experiments_integration = self.raw_yaml["experiments_integration"]

    def _find_matching_files(self, directory: Path, prefix: str) -> list[str]:
        """
        주어진 디렉토리에서 특정 접두사를 가진 파일 목록을 반환.

        Args:
            directory (Path): 탐색할 디렉토리 경로
            prefix (str): 파일 이름의 접두사

        Returns:
            list[str]: 접두사와 일치하는 파일 이름 목록
        """
        pattern = re.compile(rf"^{prefix}_v\d+\.\d+\.\d+\.csv$")
        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path '{directory}' is not a directory.")

        # 정규표현식을 사용해 접두사와 일치하는 파일 필터링
        return [f.name for f in directory.iterdir() if f.is_file() and pattern.match(f.name)]

    def _find_latest_file(self, directory: Path, prefix: str) -> tuple[Path, str]:
        """
        디렉토리 내 접두사와 일치하는 파일 중 가장 최신 파일과 버전을 반환.

        Args:
            directory (Path): 탐색할 디렉토리 경로
            prefix (str): 파일 이름의 접두사

        Returns:
            tuple[Path, str]: 최신 파일 경로와 최신 버전
        """
        file_list = self._find_matching_files(directory, prefix)
        if not file_list:
            raise FileNotFoundError(f"No files matching prefix '{prefix}' found in '{directory}'.")

        # 시맨틱 버전 기준으로 파일 정렬
        file_list.sort(key=lambda x: Version(re.search(r"v(\d+\.\d+\.\d+)", x).group(1)), reverse=True)
        latest_file = file_list[0]
        latest_version = re.search(r"v(\d+\.\d+\.\d+)", latest_file).group(1)

        return directory / latest_file, latest_version

    def _update_version(self, directory: Path, prefix: str) -> Path:
        """
        최신 파일을 탐색하고 설정 파일(`data_version.yaml`)을 업데이트.

        Args:
            directory (Path): 탐색할 디렉토리 경로
            prefix (str): 파일 이름의 접두사

        Returns:
            Path: 최신 파일 경로
        """
        latest_file, version = self._find_latest_file(directory, prefix)
        configured_version = self.latest_version[prefix]

        # 발견된 버전이 설정된 최신 버전보다 이전인 경우 에러 발생
        if Version(version) < Version(configured_version):
            raise ValueError(
                f"스캔된 {prefix} 데이터 버전 {version}이 설정된 최신 버전 {configured_version}보다 이전입니다. "
                f"데이터 디렉토리를 확인하거나 설정을 업데이트하세요."
            )

        # 발견된 버전이 설정된 최신 버전보다 새로운 경우 업데이트
        if Version(version) > Version(configured_version):
            self.latest_version[prefix] = version
            yaml_key = f"latest_{prefix}_version"
            self.raw_yaml[yaml_key] = version

            # YAML 파일 저장
            with self.data_version_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.raw_yaml, f, default_flow_style=False, allow_unicode=True)

        return latest_file

    def search_latest_train_data(self) -> pd.DataFrame:
        """
        최신 학습 데이터를 로드.

        Returns:
            pd.DataFrame: 최신 학습 데이터
        """
        latest_file = self._update_version(self.data_path, "train")
        return pd.read_csv(latest_file)

    def search_latest_valid_data(self) -> pd.DataFrame:
        """
        최신 검증 데이터를 로드.

        Returns:
            pd.DataFrame: 최신 검증 데이터
        """
        latest_file = self._update_version(self.data_path, "valid")
        return pd.read_csv(latest_file)

    def search_latest_test_data(self) -> pd.DataFrame:
        """
        최신 테스트 데이터를 로드.

        Returns:
            pd.DataFrame: 최신 테스트 데이터
        """
        latest_file = self._update_version(self.data_path, "test")
        return pd.read_csv(latest_file)

    def search_latest_experiments_data(self) -> dict[int, pd.DataFrame]:
        """
        실험 데이터에서 주요 버전별 최신 데이터를 로드하고 YAML 파일에 반영.

        Returns:
            dict[int, pd.DataFrame]: 주요 버전별 데이터프레임 딕셔너리
        """
        version_pattern = re.compile(r"v(\d+\.\d+\.\d+)")
        versions = {}

        # 실험 데이터 디렉토리에서 최신 버전 탐색
        for file in self.experiment_data_path.iterdir():
            if file.is_file():
                match = version_pattern.search(file.name)
                if match:
                    version_str = match.group(1)
                    version_obj = Version(version_str)
                    major = version_obj.major

                    # 같은 Major 버전 중 가장 최신 버전만 유지
                    if major not in versions or Version(versions[major][1]) < version_obj:
                        versions[major] = (file, version_str)

        # 새로운 Major 버전이 발견되면 self.latest_version["exp"]에 추가
        for major, (_, version_str) in versions.items():
            if major not in self.latest_version["exp"] or Version(version_str) > Version(
                self.latest_version["exp"].get(major, "0.0.0")
            ):
                self.latest_version["exp"][major] = version_str

        # 설정 파일 업데이트
        self.raw_yaml["latest_experiments_version"] = self.latest_version["exp"]
        with self.data_version_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw_yaml, f, default_flow_style=False, allow_unicode=True)

        # 최신 데이터를 로드하여 반환
        return {major: pd.read_csv(file) for major, (file, _) in versions.items()}

    def search_experiments_integration_data(self) -> pd.DataFrame:
        """
        통합된 실험 데이터를 로드.

        Returns:
            pd.DataFrame: 통합 실험 데이터
        """
        matching_files = [
            f for f in self.experiment_data_path.iterdir() if self.experiments_integration in f.name and f.is_file()
        ]
        if not matching_files:
            raise FileNotFoundError("No file with 'integration' in its name was found.")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files with 'integration' in their names were found: {matching_files}")

        return pd.read_csv(matching_files[0])
