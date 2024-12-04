import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import CHOICES, CRAWLED_TEXT, DOCUMENT, KEYWORD_PREFIX, PAGE_SUFFIX, PARAGRAPH, QUESTION, QUESTION_PLUS
from .langchain_manager import LangchainManager


class ParagraphGenerator(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")

        # 출력이 문자열인 prompts/templates/paragraph_generation/document_summarization.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="generation_from_wiki.txt",
            output_type="str",
        )

    def process(self):
        tqdm.pandas()
        columns_to_join = [KEYWORD_PREFIX + str(i) + PAGE_SUFFIX for i in range(1, 6)]
        self.source_data[CRAWLED_TEXT] = self.source_data[columns_to_join].apply(lambda row: "\n".join(row), axis=1)
        self.source_data = self.source_data[[PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, CRAWLED_TEXT]]

        self.source_data[DOCUMENT] = self.source_data.progress_apply(
            lambda row: self.lanchain_manager.invoke(row.to_dict()), axis=1
        )
