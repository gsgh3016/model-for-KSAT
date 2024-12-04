import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from prompts import load_template

from .text_crawler import WikipediaCrawler

PARAGRAPH = "paragraph"
QUESTION_PLUS = "question_plus"
QUESTION = "question"
CHOICES = "choices"


class ParagraphGenerator:
    def __init__(self):
        """
        환경 변수를 로드하고, 템플릿을 불러와 GPT 모델과 연결한 후,
        위키피디아 텍스트 크롤러를 초기화하는 생성자입니다.
        """
        load_dotenv()

        # 목적에 맞는 다른 체인 설정
        self.generation_chain = self._build_chain("generation")
        self.triming_chain = self._build_chain("triming")

        # 위키피디아 크롤러 초기화
        self.text_crawler = WikipediaCrawler()

    def _build_chain(self, chain_type: str) -> RunnableSerializable[dict, BaseMessage]:
        """
        `chain_type`을 따라 체인을 빌드

        Args:
            chain_type (str): 빌드할 체인 종류

        Returns:
            RunnableSerializable: 빌드한 체인

        Raises:
            TypeError: `chain_type`에 `"generation"` 혹은 `"triming"`을 넣지 않은 경우 예외 발생
        """
        # chain_type 체크
        try:
            if chain_type == "generation":
                file_name = "generation_from_wiki.txt"
            elif chain_type == "triming":
                file_name = "triming_paragraph.txt"
            else:
                raise TypeError('chain_type에 "generation"` 혹은 "triming"을 넣어 주세요.')
        except TypeError as e:
            raise e

        # 위키피디아 텍스트를 기반으로 지문 생성을 위한 프롬프트 템플릿 로드
        prompt = PromptTemplate.from_template(
            template=load_template(file_name=file_name, template_type="paragraph_generation")
        )

        # GPT 모델 설정 (gpt-4o-mini 모델 사용)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        chain = prompt | llm
        return chain

    def generate_raw_paragraph(self, row: pd.Series, keywords: list[str]) -> str:
        """
        `keywords` 배열에 담긴 내용을 바탕으로 위키피디아 텍스트를 크롤링하여 지문을 생성합니다.
        지문 생성 시 문제가 필요한 정보를 포함하도록 합니다.

        Args:
            problem (str): 기존 문제
            keywords (list[str]): 검색할 위키피디아 텍스트 들

        Returns:
            str: 생성된 지문
        """
        crawled_text = ""
        for keyword in keywords:
            # 위키피디아에서 해당 키워드로 텍스트 크롤링
            crawled_text += self.text_crawler.crawl_text(keyword=keyword) + "\n"

        # GPT 모델을 사용해 생성된 지문을 반환
        paragraph = self.generation_chain.invoke(
            {
                "crawled_text": crawled_text,
                PARAGRAPH: row[PARAGRAPH],
                QUESTION_PLUS: row[QUESTION_PLUS],
                QUESTION: row[QUESTION],
                CHOICES: row[CHOICES],
            }
        ).content
        return paragraph

    def trim_raw_paragraph(self, row: pd.Series) -> str:
        """
        `generate_raw_paragraph`에서 생성된 지문을 수능 지문으로 다듬습니다.

        Args:
            problem (str): 기존 문제
            raw_paragraph (str): 다듬기 전 지문

        Returns:
            str: 생성된 지문
        """
        # GPT 모델을 사용해 생성된 지문을 반환
        paragraph = self.triming_chain.invoke(
            {
                "raw_paragraph": row["1st_generated_paragraph"],
                PARAGRAPH: row[PARAGRAPH],
                QUESTION_PLUS: row[QUESTION_PLUS],
                QUESTION: row[QUESTION],
                CHOICES: row[CHOICES],
            }
        ).content
        return paragraph
