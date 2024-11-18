import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


# 데이터 요약 출력 함수
def display_data_summary(df: pd.DataFrame):
    st.subheader("데이터 요약")
    summary = pd.DataFrame(
        {
            "Total Data": df.count() + df.isnull().sum(),
            "Non-Null Count": df.count(),
            "Null Count": df.isnull().sum(),
            "Data Type": df.dtypes,
        }
    )
    st.dataframe(summary)

    st.subheader("데이터 샘플")
    st.write(df.head())


# 인덱스 접근 함수
def access_data_by_index(df: pd.DataFrame):
    st.markdown("#### Access Data by Index")
    index_input = st.number_input(
        "Enter the index of the row to retrieve:",
        min_value=0,
        max_value=len(df) - 1,
        step=1,
        key="unique_key_1",
    )
    if st.button("Retrieve by Index"):
        if 0 <= index_input < len(df):
            row_data = df.iloc[int(index_input)]
            st.write(f"Row at index {int(index_input)}:")
            st.write(row_data)
        else:
            st.error("Invalid index. Please try again.")


# 칼럼 필터링 함수
def filter_data_by_column(df: pd.DataFrame):
    st.markdown("#### Filter Data by Column")
    column = st.selectbox("Select a column to filter by:", df.columns)
    search_value = st.text_input(f"Enter the value to search in '{column}':")

    if st.button("Search"):
        filtered_df = df[df[column].astype(str).str.contains(search_value, na=False, case=False, regex=False)]
        result_count = len(filtered_df)
        st.write(f"Number of rows containing '{search_value}' in column '{column}': {result_count}")
        if result_count > 0:
            st.dataframe(filtered_df)
        else:
            st.write("No matching data found.")


# 수능 형식으로 데이터 출력해주는 함수
def display_question_format(df: pd.DataFrame):
    st.subheader("문제 형태로 확인")
    required_columns = {"paragraph", "question", "choices"}
    if not required_columns.issubset(df.columns):
        st.error("The uploaded file must contain the following columns: paragraph, question, choices, answer")
    else:
        question_idx = st.number_input(
            "Enter the index of the row to retrieve:",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
            key="unique_key_2",
        )
        row = df.iloc[question_idx]
        paragraph = row["paragraph"]
        question = row["question"]
        choices = row["choices"]

        st.markdown("#### 📜 지문")
        st.write(paragraph)
        st.markdown("#### ❓ 문제")
        st.write(question)
        if "question_plus" in df.columns and not pd.isnull(row["question_plus"]):
            st.markdown(body="#### 🔍 <보기>")
            st.write(row["question_plus"])

        choices_list = eval(choices) if isinstance(choices, str) else choices
        st.markdown("#### 📝 선택지")
        for idx, choice in enumerate(choices_list, start=1):
            st.write(f"{idx}. {choice.strip()}")
        if "answer" in df.columns:
            st.markdown("#### ✅ 정답")
            st.write(row["answer"])


if __name__ == "__main__":

    # 페이지 기본 설정
    st.set_page_config(page_title="Data Analysis Dashboard", layout="wide", page_icon="📊")

    # 사이드바 메뉴 설정
    with st.sidebar:
        st.image(
            "assets/pikachu-boxing.gif",
            width=128,
        )  # 로고
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
