from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SpecialTokens:
    """
    스페셜 토큰 관리용 데이터 클래스.
    """

    start_of_response: str = "<response_start>"
    start_of_reasoning: str = "<reasoning_start>"
    end_of_reasoning: str = "<reasoning_end>"
    start_of_answer: str = "<answer_start>"
    end_of_answer: str = "<answer_end>"
    end_of_response: str = "<response_end>"

    @staticmethod
    def to_list() -> list[str]:
        """
        모든 스페셜 토큰을 리스트로 반환합니다.

        Returns:
            List[str]: 스페셜 토큰 리스트.
        """
        return [getattr(SpecialTokens, field.name) for field in fields(SpecialTokens)]


if __name__ == "__main__":
    print(SpecialTokens.to_list())
