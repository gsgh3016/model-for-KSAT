import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import DOCUMENT, KEYWORD_PREFIX, KEYWORDS, PAGE_SUFFIX, SUMMARY_SUFFIX
from .langchain_manager import LangchainManager


class Summarizer(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if data:
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")

        # 출력이 문자열인 prompts/templates/paragraph_generation/document_summarization.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="document_summarization.txt",
            output_type="str",
        )

    def summarize(self, row: pd.Series, i: int) -> str:
        """
        위키피디아 문서를 요약하는 함수

        Args:
            row (pd.Series): 제공된 데이터셋의 한 행(row), 문제 하나
            i {int}: 참고할 키워드

        Returns:
            str: 추출된 5개 키워드
        """

        keywords_list = [row[KEYWORD_PREFIX + str(idx)] for idx in range(1, 6)]
        keywords = ", ".join(keywords_list)

        if pd.isna(row[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX]):
            return ""
        document = row[KEYWORD_PREFIX + str(i) + PAGE_SUFFIX]

        response = self.lanchain_manager.invoke(
            {
                KEYWORDS: keywords,
                DOCUMENT: document,
            },
        )
        return response

    def process(self):
        tqdm.pandas()
        for i in range(1, 6):
            self.source_data[KEYWORD_PREFIX + str(i) + SUMMARY_SUFFIX] = self.source_data.progress_apply(
                lambda row: self.summarize(row, i), axis=1
            )
