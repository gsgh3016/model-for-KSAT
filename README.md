# 수능 문제 풀이 모델

한국어의 특성과 수능 시험의 특징을 바탕으로 수능에 최적화된 모델을 만듭니다.

## Getting Started

### Requirements

- Python: 3.10
- CUDA: >= 12.1
- PyTorch: 2.5.1+cu121

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
