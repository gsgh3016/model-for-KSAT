import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

from streamlit_utils import access_data_by_index, display_data_summary, display_question_format, filter_data_by_column

if __name__ == "__main__":

    # 페이지 기본 설정
    st.set_page_config(page_title="Data Analysis Dashboard", layout="wide", page_icon="📊")

    # 사이드바 메뉴 설정
    with st.sidebar:
        st.title("Analysis Dashboard")
        selected = option_menu(
            "Main Menu", ["Home", "Compare"], icons=["house", "arrows-expand"], menu_icon="menu", default_index=0
        )

    # HOME 탭
    if selected == "Home":
        st.title("📊 Data Analysis")
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type="csv")
        tab1, tab2, tab3 = st.tabs(["📊 데이터 개요", "🔍 데이터 탐색", "📈 데이터 분포"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            # 첨부 파일이 없으면 기본적으로 train.csv에 대한 분석을 출력합니다.
            df = pd.read_csv("data/train.csv")
        # 데이터 요약
        with tab1:
            display_data_summary(df)

        # 개별 데이터 접근
        with tab2:
            st.subheader("전체 데이터 확인")
            st.dataframe(df)

            st.subheader("개별 데이터 확인")
            access_method = st.radio("데이터 접근 방식 선택", ("Access by Index", "Filter by Column"))
            if access_method == "Access by Index":
                access_data_by_index(df)
            elif access_method == "Filter by Column":
                filter_data_by_column(df)

            display_question_format(df)

        # 분포 확인
        with tab3:
            st.subheader("데이터 분포")
            if df is not None:
                pass  # TODO: Add distribution plotting logic
            else:
                st.write("Please upload a CSV file to view the analytics.")
