import os

import pandas as pd
import wikipediaapi as wk
from dotenv import load_dotenv
from tqdm import tqdm

from .constants import EXISTS_SUFFIX, KEYWORD_PREFIX, PAGE_SUFFIX


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

    @property
    def data(self) -> pd.DataFrame:
        return self.data

    def crawl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        위키피디아에 문서가 존재하는 여부를 판별하는 함수입니다.

        Args:
            data (pd.DataFrame): keyword_i 열이 있는 데이터 프레임

        Returns:
            pd.DataFrame: 위키피디아 문서 존재 여부와 문서 텍스트 내용이 저장된 데이터
        """
        for i in range(1, 6):
            data[KEYWORD_PREFIX + str(i) + EXISTS_SUFFIX] = False
            data[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX] = ""

        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Crawling Wikipedia"):
            for j in range(1, 6):
                page = self.wiki.page(row[KEYWORD_PREFIX + str(j)])
                data.at[idx, KEYWORD_PREFIX + str(j) + EXISTS_SUFFIX] = page.exists()
                data.at[idx, KEYWORD_PREFIX + str(j) + PAGE_SUFFIX] = page.text
        self.data = data
