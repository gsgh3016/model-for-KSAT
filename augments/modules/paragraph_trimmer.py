import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from .constants import ANSWER, CHOICES, DOCUMENT, PARAGRAPH, QUESTION, QUESTION_PLUS, RAW_PARAGRAPH
from .langchain_manager import LangchainManager


class ParagraphTrimmer(BaseProcessor):
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/", data: pd.DataFrame = None):
        super().__init__(data_path, experiment_data_path)

        if isinstance(data, pd.DataFrame):
            self.source_data = data
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be pandas.DataFrame format")

        # 출력이 문자열인 prompts/templates/paragraph_generation/document_summarization.txt 기반 체인 설정
        self.lanchain_manager = LangchainManager(
            prompt_type="paragraph_generation",
            prompt_source="trimming_paragraph.txt",
            output_type="str",
        )

    def process(self):
        tqdm.pandas()
        df = self.source_data[[PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, RAW_PARAGRAPH]]

        self.source_data[DOCUMENT] = df.progress_apply(lambda row: self.lanchain_manager.invoke(row.to_dict()), axis=1)

        self.source_data[QUESTION_PLUS] = self.source_data.apply(
            lambda row: row[PARAGRAPH] if pd.isna(row[QUESTION_PLUS]) else row[PARAGRAPH] + "\n" + row[QUESTION_PLUS],
            axis=1,
        )
        self.source_data[PARAGRAPH] = self.source_data[DOCUMENT]

        self.source_data = self.source_data[[PARAGRAPH, QUESTION_PLUS, QUESTION, CHOICES, ANSWER]]
