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
        self.experiments_integration_version = self.raw_yaml["experiments_integration_version"]

    def update_file_path(
        self,
        major: int,
        minor: int,
        update_target: str = "micro",
        save_in_experiment: bool = False,
    ) -> Path:
        """
        특정 Major, Minor 버전에 대해 새로운 파일 경로를 생성하여 반환.

        Args:
            major (int): Major 버전.
            minor (int): Minor 버전.
            update_target (str): 업데이트 대상 ("major", "minor", "micro").
            save_in_experiment (bool): 실험 데이터 경로에 저장 여부.

        Returns:
            Path: 새로 생성된 파일 경로.
        """
        # 인자 검사
        self._validate_update_target(update_target)

        # 저장 디렉토리 설정
        dir = self._get_save_directory(save_in_experiment)

        # 최신 파일 검색
        latest_file = self._find_latest_file_in_directory(dir, major, minor)

        # 새로운 버전 생성
        new_version = self._generate_new_version(latest_file, update_target)

        # 새로운 파일 이름 생성 및 경로 반환
        return self._create_new_file_path(latest_file, new_version, dir)

    def _validate_update_target(self, update_target: str):
        if update_target not in ["minor", "major", "micro"]:
            raise ValueError('update_target에 "major", "minor", "micro" 로만 입력하세요.')

    def _get_save_directory(self, save_in_experiment: bool) -> Path:
        return Path(self.experiment_data_path if save_in_experiment else self.data_path)

    def _find_latest_file_in_directory(self, dir: Path, major: int, minor: int) -> Path:
        latest_file = None
        version_pattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
        for file in dir.iterdir():
            if file.is_file():
                match = version_pattern.search(file.name)
                if match:
                    version_obj = Version(match.group(0))
                    if version_obj.major == major and version_obj.minor == minor:
                        if (
                            latest_file is None
                            or Version(latest_file.name.split("_v")[-1].split(".csv")[0]) <= version_obj
                        ):
                            latest_file = file

        # Major.Minor 버전 파일이 없으면 새 파일 생성
        if not latest_file:
            print(f"{dir}에 Major {major}, Minor {minor} 버전에 해당하는 파일이 없습니다. 새로운 버전을 생성합니다.")
            latest_file = self._create_placeholder_file(dir, major, minor)

        return latest_file

    def _create_placeholder_file(self, dir: Path, major: int, minor: int) -> Path:
        """
        Major.Minor 버전에 해당하는 새로운 기본 파일 생성

        Args:
            dir (Path): 저장 디렉토리
            major (int): Major 버전
            minor (int): Minor 버전

        Returns:
            Path: 생성된 파일 경로
        """
        new_version = f"v{major}.{minor}.0"
        file_name = f"data_{new_version}.csv"
        file_path = dir / file_name
        file_path.touch()  # 빈 파일 생성
        return file_path

    def _generate_new_version(self, latest_file: Path, update_target: str) -> str:
        version_pattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
        match = version_pattern.search(latest_file.name)
        if not match:
            raise ValueError(f"파일 이름에서 버전을 찾을 수 없습니다: {latest_file.name}")

        version = Version(match.group(0))

        if update_target == "major":
            new_major, new_minor, new_micro = version.major + 1, 0, 0
        elif update_target == "minor":
            new_major, new_minor, new_micro = version.major, version.minor + 1, 0
        elif update_target == "micro":
            new_major, new_minor, new_micro = version.major, version.minor, version.micro + 1

        return f"v{new_major}.{new_minor}.{new_micro}"

    def _create_new_file_path(self, latest_file: Path, new_version: str, dir: Path) -> Path:
        version_pattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
        match = version_pattern.search(latest_file.name)

        prefix = latest_file.name[: match.start()]
        suffix = latest_file.name[match.end() :]
        file_name = prefix + new_version + suffix

        return dir / file_name

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

    def _update_experiments_version(self, latest_version_obj: Version, version_type: str, major: int):
        """
        실험 데이터 버전 정보를 업데이트 하는 함수

        Args:
            latest_version_obj (Version): 최신 버전
            version_type (str): 데이터 통합 이후 이전 여부
            major (int): Major 버전 정보
        """
        experiments_integrations: dict[int:str] = self.latest_version["exp"]
        if version_type not in experiments_integrations.keys():
            raise TypeError("yaml 파일 설정에 맞춰 version_type을 넣어주세요.")

        updated_version = Version(
            f"{latest_version_obj.major}.{latest_version_obj.minor}.{latest_version_obj.micro + 1}"
        )
        experiments_integrations[version_type][major] = str(updated_version)
        self.raw_yaml["latest_experiments_version"] = experiments_integrations
        with self.data_version_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw_yaml, f, default_flow_style=False, allow_unicode=True)

    def _validate_version(self, major: int, is_experiment: bool = False) -> Version:
        # Major 버전이 통합된 기준보다 낮으면 통합 후 버전으로 설정
        version_type = "after_integration" if is_experiment else "before_integration"
        versioning: dict[int, str] = self.latest_version["exp"][version_type]
        if major not in versioning.keys():
            raise KeyError(f"{major}.x.x 버전은 latest_experiments_version.{version_type}에 존재하지 않습니다.")

        # 최신 버전 정보를 가져오고 객체로 변환
        return Version(versioning[major])

    def search_latest_experiments_data(self, major: int, minor: int, is_experiment: bool = False) -> pd.DataFrame:
        """
        특정 Major 버전의 실험 데이터에서 최신 데이터를 로드하고, 필요 시 Patch를 +1하여 YAML 파일에 업데이트.

        Args:
            major (int): 검색할 Major 버전
            minor (int): 검색할 Minor 버전
            is_experiment (bool): True인 경우, 최신 버전의 Patch를 +1하고 YAML 파일에 반영

        Returns:
            pd.DataFrame: 지정된 Major 버전의 최신 실험 데이터
        """
        latest_version_obj = self._validate_version(major=major, is_experiment=is_experiment)
        version_pattern = re.compile(r"v\d+\.\d+\.\d+")

        # 디렉토리에서 파일 검색 및 최신 버전 찾기
        latest_file = None
        for file in self.experiment_data_path.iterdir():
            if file.is_file():
                match = version_pattern.search(file.name)
                if match:
                    version_obj = Version(match.group(0))
                    if version_obj.major == major and version_obj.minor == minor and version_obj >= latest_version_obj:
                        latest_version_obj = version_obj
                        latest_file = file

        if not latest_file:
            raise FileNotFoundError(
                f"{'통합 후' if is_experiment else '통합 전'} Major 버전 {major}에 해당하는 실험 데이터가 존재하지 않습니다."
            )

        # is_experiment가 True인 경우 Patch를 +1하여 YAML 업데이트
        version_type = "after_integration" if is_experiment else "before_integration"
        if is_experiment:
            self._update_experiments_version(
                latest_version_obj=latest_version_obj, version_type=version_type, major=major
            )

        # 최신 데이터를 로드하여 반환
        return pd.read_csv(latest_file)

    def get_latest_experiment_data_path(self, major: int, is_experiment: bool = False) -> Path:
        """
        특정 Major 버전에 해당하는 최신 실험 데이터 경로를 반환.

        Args:
            major (int): 검색할 Major 버전
            is_experiment (bool): 호출하는 시점이 실험 이후인지 설정. 기본값은 `False`

        Returns:
            Path: 최신 실험 데이터 파일 경로
        """
        latest_version_obj = self._validate_version(major=major, is_experiment=is_experiment)
        version_pattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)")

        latest_file = None
        # 실험 데이터 경로의 파일들을 순회하며 최신 버전에 해당하는 파일 검색
        for file in self.experiment_data_path.iterdir():
            if file.is_file():
                match = version_pattern.search(file.name)
                if match:
                    version_obj = Version(match.group(0))
                    # Major 버전과 버전 객체가 모두 일치하는 파일을 찾으면 해당 파일을 반환
                    if version_obj.major == major and version_obj == latest_version_obj:
                        latest_file = file
                        break

        # 최신 파일이 없을 경우, 적절한 에러 메시지와 함께 예외 발생
        if not latest_file:
            raise FileNotFoundError(
                f"{'통합 후' if is_experiment else '통합 전'} Major 버전 {major}에 해당하는 최신 실험 데이터가 존재하지 않습니다."
            )

        return latest_file

    def search_experiments_integration_data(self, is_experiment: bool = False) -> pd.DataFrame:
        """
        통합된 실험 데이터를 로드하거나, 필요 시 Patch 버전을 +1하여 YAML 파일에 업데이트.

        Args:
            is_experiment (bool): True인 경우, 3.0.x 데이터의 Patch 버전을 +1하고 YAML 파일에 반영

        Returns:
            pd.DataFrame: 최신 통합 실험 데이터
        """
        # 통합 데이터 파일 검색
        matching_files = [
            f for f in self.experiment_data_path.iterdir() if self.experiments_integration in f.name and f.is_file()
        ]

        if not matching_files:
            raise FileNotFoundError("No file with 'integration' in its name was found.")
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files with 'integration' in their names were found: {matching_files}")

        # 파일명 추출 및 최신 버전 확인
        version_type = "after_integration"
        major = 3
        latest_version_str = self.latest_version["exp"][version_type].get(major, f"{major}.0.0")
        latest_version_obj = Version(latest_version_str)

        # is_experiment가 True인 경우 Patch를 +1하여 YAML 업데이트
        if is_experiment:
            self._update_experiments_version(
                latest_version_obj=latest_version_obj, version_type=version_type, major=major
            )

        # 통합 데이터 로드
        integration_file = matching_files[0]
        return pd.read_csv(integration_file)
