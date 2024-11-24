from ast import literal_eval
from typing import List, Union

import matplotlib.axes._axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_before_length_plot(df: pd.DataFrame):
    # 길이 표출을 위한 결측값 처리
    df["question_plus"].fillna("", inplace=True)

    # 표출할 columns와 적절한 bin_sizes
    columns = ["paragraph", "question", "choices", "question_plus"]
    bin_sizes = [100, 10, 30, 10, 50]
    return df, columns, bin_sizes


def plot_length_distribution_percentage(
    ax: axes.Axes, df: pd.DataFrame, column_name: Union[str, List[str]], bin_size=10, color="skyblue"
):
    """
    주어진 DataFrame의 column 문자열 길이 분포를 히스토그램으로 시각화하며, 각 bin의 비율(%)을 계산하여 표시합니다.
    """
    # 표출할 column이 list일 경우 총계로 계산, title에 기입될 column_name을 total로 수정
    if isinstance(column_name, list):
        lengths = df[column_name].map(len).sum(axis=1)
        column_name = "total"
    else:
        lengths = df[column_name].map(len)

    # Bin 범위 설정
    max_length = lengths.max()
    bins = range(0, max_length + bin_size, bin_size)

    # 히스토그램 계산
    counts, edges = np.histogram(lengths, bins=bins)

    # 각 bin의 비율(%) 계산
    percentages = (counts / len(lengths)) * 100

    ax.bar(edges[:-1], percentages, width=bin_size, color=color, edgecolor="black", alpha=0.7, align="edge")
    ax.set_title(f"{column_name} length distribution", fontsize=16)
    ax.set_xlabel("Length of " + column_name, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def column_length_distribution(df: pd.DataFrame):
    df, columns, bin_sizes = set_before_length_plot(df)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, column, bin_size in zip(axes, columns, bin_sizes):
        plot_length_distribution_percentage(ax=ax, df=df, column_name=column, bin_size=bin_size)

    fig.tight_layout()
    return fig


def total_length_distribution(df: pd.DataFrame):
    df, columns, bin_sizes = set_before_length_plot(df)

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_length_distribution_percentage(ax=ax, df=df, column_name=columns, bin_size=bin_sizes[-1])

    fig.tight_layout()
    return fig


def plot_choices_length_distribution(ax: axes.Axes, df: pd.DataFrame, bins=2, color="skyblue"):
    df["choices"] = df["choices"].apply(literal_eval)
    list_lengths = df["choices"].apply(len)
    # 빈도수 계산 및 비율로 변환
    value_counts = list_lengths.value_counts(normalize=True).sort_index() * 100

    # 비율 text 명시
    for idx, value in zip(value_counts.index, value_counts.values):
        ax.text(idx, value - 5, f"{value:.1f}%", ha="center", fontsize=10, color="black")

    ax.bar(value_counts.index, value_counts.values, color=color, edgecolor="black", alpha=0.7)
    ax.set_title("Choices Length Distribution", fontsize=16)
    ax.set_xlabel("Length of Choices", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(value_counts.index)


def choices_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    plot_choices_length_distribution(ax, df)

    fig.tight_layout()
    return fig
