import pandas as pd

# TODO: 기능 통합 시, 아래 함수를 키워드 추출 모듈과 유형 분류 모듈이 동작하고 사용할 것


def integrate_data_by_row(type_series: pd.Series, keyword_series: pd.Series) -> pd.Series:
    """
    데이터 열에서 겹치는 행을 제외한 나머지 행을 이어붙이는 함수입니다.

    Args:
        type_series (pd.Series): 문제 유형을 분류한 데이터의 단일 열
        keyword_series (pd.Series): 키워드를 추출한 데이터의 단일 열

    Returns:
        pd.Series: 이어붙인 결과
    """
    return type_series.combine(
        keyword_series, lambda x, y: (x if pd.isna(y) else y if pd.isna(x) else x if x == y else f"{x}, {y}")
    )


def integrate_data(type_df: pd.DataFrame, keyword_df: pd.DataFrame) -> pd.DataFrame:
    """
    문제 유형 분류 실험 결과 데이터와 키워드 추출 실험 결과 데이터를 통합하는 함수입니다.

    Args:
        type_df (pd.DataFrame): 문제 유형 분류 결과 데이터
        keyword_df (pd.DataFrame): 키워드 유형 분류 결과 데이터

    Returns:
        pd.DataFrame: 통합한 데이터
    """
    # 병합: 'id' 열을 기준으로 병합
    merged_df = pd.merge(type_df, keyword_df, on="id", how="outer", suffixes=("_type", "_keyword"))

    # 열 통합
    for col in [col for col in type_df.columns if col != "id"]:
        if col in keyword_df.columns:
            merged_df[col] = integrate_data_by_row(merged_df[f"{col}_type"], merged_df[f"{col}_keyword"])

    # 불필요한 열 제거
    merged_df.drop(columns=[col for col in merged_df.columns if col.endswith(("_type", "_keyword"))], inplace=True)

    # 인덱스 초기화
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df
