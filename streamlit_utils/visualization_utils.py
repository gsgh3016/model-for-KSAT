import matplotlib.axes._axes as axes
import numpy as np
import pandas as pd


def plot_length_distribution_percentage(ax: axes.Axes, df: pd.DataFrame, column: str, bin_size=10, color="skyblue"):
    """
    주어진 DataFrame의 column 문자열 길이 분포를 히스토그램으로 시각화하며, 각 bin의 비율(%)을 계산하여 표시합니다.
    """
    # 문자열 길이 계산
    lengths = df[column].str.len()

    # Bin 범위 설정
    max_length = lengths.max()
    bins = range(0, max_length + bin_size, bin_size)

    # 히스토그램 계산
    counts, edges = np.histogram(lengths, bins=bins)

    # 각 bin의 비율(%) 계산
    percentages = (counts / len(lengths)) * 100

    ax.bar(edges[:-1], percentages, width=bin_size, color=color, edgecolor="black", alpha=0.7, align="edge")
    ax.set_title(f"{column} Length Distribution (Percentage)", fontsize=16)
    ax.set_xlabel("Length of " + column, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
