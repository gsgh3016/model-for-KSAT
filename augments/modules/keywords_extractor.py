import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import (
    ANSWER,
    CATEGORY,
    CHOICES,
    KEYWORD_PREFIX,
    KEYWORDS,
    NEED_KNOWLEDGE,
    PARAGRAPH,
    QUESTION,
    QUESTION_PLUS,
    REASONING,
)
from .langchain_manager import LangchainManager


class KeywordsExtractor(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")

        # 출력이 문자열인 prompts/templates/keyword_extraction/wikipedia_search_keyword_with_reasoning.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(
            prompt_type="keyword_extraction",
            prompt_source="wikipedia_search_keyword_with_reasoning.txt",
            output_type="str",
        )

    def extract_keyword(self, data: pd.Series) -> list[str]:
        """
        문제 해결에 필요한 키워드 5개를 추출하고 결과를 배열로 반환하는 함수

        Args:
            data (pd.Series): 제공된 데이터셋의 한 행(row), 문제 하나

        Returns:
            list[str]: 추출된 5개 키워드
        """
        response = self.lanchain_manager.invoke(data.to_dict())
        response_split = response.split(",")
        keywords = [keyword.strip() for keyword in response_split]
        return keywords

    def process(self):
        tqdm.pandas()
        self.source_data = self.source_data[self.source_data[CATEGORY] == NEED_KNOWLEDGE]
        self.source_data = self.source_data[[PARAGRAPH, QUESTION, QUESTION_PLUS, CHOICES, ANSWER, REASONING]]

        # extract_keyword 함수로 키워드를 추출
        self.source_data[KEYWORDS] = self.source_data.progress_apply(self.extract_keyword, axis=1)

        # 키워드를 유용하게 다루기 위해 키워드 별로 칼럼으로 분리
        keywords_expanded = pd.DataFrame(self.source_data[KEYWORDS].tolist(), index=self.source_data.index)
        keywords_expanded.columns = [KEYWORD_PREFIX + str(i + 1) for i in range(keywords_expanded.shape[1])]

        # 기존 데이터를 키워드 칼럼들과 병합
        self.source_data = pd.concat([self.source_data.drop(columns=[KEYWORDS]), keywords_expanded], axis=1)
