import os

import wikipediaapi as wk
from dotenv import load_dotenv


class WikipediaCrawler:
    def __init__(self, languange="ko"):
        """
        위키피디아에서 텍스트를 크롤링하는 모듈입니다.

        Args:
            languange (str, optional): 위키피디아 언어를 설정합니다.
            기본 값은 "ko"이며, https://en.wikipedia.org/wiki/List_of_Wikipedias#Active_edition를 참고해서 인자를 수정해주세요.
        """
        load_dotenv()
        USER_AGENT = os.getenv("WIKI_USER_AGENT")
        self.wiki = wk.Wikipedia(user_agent=USER_AGENT, language=languange)

    def is_exists(self, keyword: str) -> tuple:
        """
        위키피디아에 `keyword` 페이지가 존재하는 여부를 판별하는 함수입니다.

        Args:
            keyword (str): 검색할 키워드

        Returns:
            tuple: 위키피디아 페이지 텍스트 존재 여부를 0번 인덱스에 답고 있는 튜플
        """
        page = self.wiki.page(keyword)
        is_exists = page.exists()
        return (is_exists, str(page.text)) if is_exists else (False,)
