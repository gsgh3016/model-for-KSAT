import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from streamlit_utils import column_length_distribution, display_data_summary, display_data_tab

if __name__ == "__main__":

    # 페이지 기본 설정
    st.set_page_config(page_title="Data Analysis Dashboard", layout="wide", page_icon="📊")

    # 사이드바 메뉴 설정
    with st.sidebar:
        st.title("Analysis Dashboard")
        selected = option_menu(
            "Main Menu", ["Home", "Compare"], icons=["house", "arrows-expand"], menu_icon="menu", default_index=0
        )
    load_dotenv()

    # HOME 탭
    if selected == "Home":
        st.title("📊 Data Analysis")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type="csv")
        experiment_file = st.sidebar.file_uploader("Upload a experiment result CSV file for analysis", type="csv")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 개요", "🔍 데이터 탐색", "📈 데이터 분포", "🔬 실험 데이터"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            # 첨부 파일이 없으면 기본적으로 설정한 학습 데이터에 대한 분석을 출력합니다.
            # .env에서 STREAMLIT_DATA_PATH, STREAMLIT_EXPERIMENT_DATA_PATH에 각각 학습 데이터, 실험 데이터를 설정하세요.
            df = pd.read_csv(os.getenv("STREAMLIT_DATA_PATH"))
        if experiment_file:
            exp_df = pd.read_csv(experiment_file)
        else:
            exp_df = pd.read_csv(os.getenv("STREAMLIT_EXPERIMENT_DATA_PATH"))

        # 데이터 요약
        with tab1:
            display_data_summary(df)

        # 개별 데이터 접근
        with tab2:
            display_data_tab(df, "tab2")

        # 분포 확인
        with tab3:
            st.subheader("컬럼 별 데이터 길이 분포")

            fig = column_length_distribution(df)
            st.pyplot(fig)

            # TODO: Add distribution plotting logic

        # 실험 데이터 확인
        with tab4:
            display_data_tab(exp_df, "tab4")

    elif selected == "Compare":
        st.title("🆚 Compare Datasets")
