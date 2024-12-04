import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from prompts import load_template

from .text_crawler import WikipediaCrawler


class ParagraphGenerator:
    def __init__(self):
        """
        환경 변수를 로드하고, 템플릿을 불러와 GPT 모델과 연결한 후,
        위키피디아 텍스트 크롤러를 초기화하는 생성자입니다.
        """
        load_dotenv()

        # 위키피디아 텍스트를 기반으로 지문 생성을 위한 프롬프트 템플릿 로드
        prompt = PromptTemplate.from_template(
            template=load_template(file_name="generation_from_wiki.txt", template_type="paragraph_generation")
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

        # 프롬프트와 LLM을 연결한 체인 생성
        self.chain = prompt | llm
        # 위키피디아 크롤러 초기화
        self.text_crawler = WikipediaCrawler()

    def generate(self, keywords: list[str]) -> str:
        """
        `keywords` 배열에 담긴 내용을 바탕으로 위키피디아 텍스트를 크롤링하여 지문을 생성합니다.

        Args:
            keywords (list[str]): 검색할 위키피디아 텍스트 들

        Returns:
            str: 생성된 지문
        """
        crawled_text = ""
        for keyword in keywords:
            # 위키피디아에서 해당 키워드로 텍스트 크롤링
            crawled_text += self.text_crawler.crawl_text(keyword=keyword) + "\n"

        # TODO: 프롬프트 설계 및 인자 넘겨주기
        # GPT 모델을 사용해 생성된 지문을 반환
        paragraph = self.chain.invoke()
        return paragraph
