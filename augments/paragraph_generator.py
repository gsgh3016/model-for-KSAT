import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from text_crawler import WikipediaCrawler

from prompts import load_template


class ParagraphGenerator:
    def __init__(self):
        load_dotenv()

        prompt = PromptTemplate.from_template(
            template=load_template(file_name="generation_from_wiki.txt", template_type="paragraph_generation")
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.chain = prompt | llm
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
            crawled_text += self.text_crawler.crawl_text(keyword=keyword) + "\n"

        # TODO: 프롬프트 설계 및 인자 넘겨주기
        paragraph = self.chain.invoke()
        return paragraph
