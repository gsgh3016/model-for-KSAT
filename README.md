# 수능 문제 풀이 모델

한국어의 특성과 수능 시험의 특징을 바탕으로 수능에 최적화된 모델을 만듭니다.

## Getting Started

### Git Clone

```shell
$ git clone git@github.com:boostcampaitech7/level2-nlp-generationfornlp-nlp-06-lv3.git
$ cd level2-nlp-generationfornlp-nlp-06-lv3
```

### Import Data

```shell
$ sudo apt-get install wget
$ wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000330/data/20241107124012/data.tar.gz
$ tar -xzvf data.tar.gz -C data
$ rm data.tar.gz
```

### Create Virtual Environment

```shell
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $
```

### Install Packages

```shell
(.venv) $ pip install -r requirements.txt
```

### Run main.py

```shell
(.venv) $ python main.py
```

## Contribution Guide

프로젝트에 기여하는 방법에 대한 가이드입니다. 아래 내용을 참고하여 작업 시 일관성을 유지해주세요.

### 커밋 템플릿 사용법

프로젝트에서 커밋 메시지 형식을 통일하기 위해 커밋 템플릿을 설정할 수 있습니다. 아래 명령어를 실행하여 템플릿을 적용하세요:

```shell
$ git config commit.template .gitcommit_template
```

- `.gitcommit_template` 파일은 프로젝트 루트에 있는 커밋 템플릿 파일입니다.
- 위 명령어를 실행하면 커밋 시 템플릿이 자동으로 불러와집니다.

### 패키지 추가 시 처리 방법

새로운 Python 패키지를 설치하거나 업데이트한 후, 프로젝트의 의존성 파일을 최신 상태로 유지하기 위해 아래 명령어를 실행하세요:

```shell
$ pip list --not-required --format=freeze > requirements.txt
```

- `pip list --not-required --format=freeze` 명령어는 현재 환경에 설치된 패키지 중 의존성 패키지가 아닌 직접 설치한 패키지들을 `requirements.txt` 파일에 기록합니다.
- 이 명령어를 통해 불필요한 패키지 정보가 포함되지 않도록 관리합니다.
- `requirements.txt` 파일은 프로젝트 실행 환경을 복제하거나 배포 시에 사용됩니다.
