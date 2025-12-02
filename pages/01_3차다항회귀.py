import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="일별 공급량 예측 (3차 다항회귀, 평균기온만)",
    layout="wide",
)

st.title("일별 공급량 예측 (평균기온 3차 다항회귀)")

# =========================
# 경로 & 상수
# =========================
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1lolxwf57MbcVTRWSR4sMTv7Z6TYelteb3BL33sPgFoo/export?format=csv&gid=0"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "daily_supply_poly3.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

target_col = "공급량(M3)"
temp_col = "평균기온"

# =========================
# 0. 데이터 로드
# =========================
@st.cache_data
def load_data_from_sheet() -> pd.DataFrame:
    df = pd.read_csv(SHEET_CSV_URL)
    return df

df_raw = load_data_from_sheet()

st.subheader("원본 데이터 (상위 10개)")
st.dataframe(df_raw.head(10))

required_cols = ["날짜", target_col, temp_col]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

# =========================
# 1. 사이드바 설정
# =========================
st.sidebar.header("모델 설정")

# 검증 비율
test_ratio = st.sidebar.slider("검증 데이터 비율(뒤쪽 기준)", 0.05, 0.3, 0.2, 0.05)

# 저장된 모델 사용 여부
saved_exists = os.path.exists(MODEL_PATH)

mode = st.sidebar.radio(
    "모드 선택",
    ("새로 학습하기", "저장된 모델 불러오기" if saved_exists else "저장된 모델 없음"),
)

save_after_train = st.sidebar.checkbox("학습 후 모델 저장", value=True)

train_button = st.sidebar.button(
    "실행 (학습 또는 불러오기)",
    type="primary"
)

# =========================
# 2. 전처리
# =========================
df_raw["날짜"] = pd.to_datetime(df_raw["날짜"])
df_raw = df_raw.sort_values("날짜").reset_index(drop=True)

def to_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

# 타깃 숫자 변환
df_raw[target_col] = to_float_series(df_raw[target_col])

# 평균기온 숫자 변환
df_raw[temp_col] = pd.to_numeric(df_raw[temp_col], errors="coerce")

# 필요한 컬럼만
df = df_raw[["날짜", temp_col, target_col]].copy()
df = df.dropna().reset_index(drop=True)

if df.empty:
    st.error("결측 제거 후 남은 데이터가 없습니다.")
    st.stop()

st.subheader("모델에 사용할 데이터 샘플")
st.write(f"사용 행 수: {len(df)}")
st.dataframe(df.head(10))

# =========================
# 3. 학습/검증 데이터 분할
#   - 시간 순서 유지, 뒤쪽을 검증으로 사용
# =========================
n = len(df)
test_size = int(n * test_ratio)
if test_size < 1:
    test_size = 1

train_df = df.iloc[:-test_size].copy()
test_df = df.iloc[-test_size:].copy()

X_train = train_df[[temp_col]].values  # (n, 1)
y_train = train_df[target_col].values

X_test = test_df[[temp_col]].values
y_test = test_df[target_col].values

train_dates = train_df["날짜"]
test_dates = test_df["날짜"]

# =========================
# 4. 3차 다항회귀 모델 정의
# =========================
def build_poly3_model():
    # 평균기온 1차, 2차, 3차 항까지 포함하는 선형회귀 파이프라인
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    return model

def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model_from_path(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# =========================
# 5. 실행 (학습 or 불러오기)
# =========================
if train_button:
    # -------------------------
    # 5-1. 저장된 모델 불러오기
    # -------------------------
    if mode == "저장된 모델 불러오기" and saved_exists:
        st.info("저장된 3차 다항회귀 모델을 불러옵니다.")
        model = load_model_from_path(MODEL_PATH)
        st.success("모델 로드 완료!")

    # -------------------------
    # 5-2. 새로 학습하기
    # -------------------------
    else:
        st.info("새 3차 다항회귀 모델을 학습합니다. (입력: 평균기온)")
        with st.spinner("모델 학습 중..."):
            model = build_poly3_model()
            model.fit(X_train, y_train)

        st.success("학습 완료!")

        # 모델 저장
        if save_after_train:
            save_model(model, MODEL_PATH)
            st.success(f"모델 저장 완료: {MODEL_PATH}")

    # -------------------------
    # 5-3. 검증 및 시각화
    # -------------------------
    # 검증 세트 예측
    y_pred = model.predict(X_test)

    # 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    non_zero_mask = y_test != 0
    if non_zero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_test[non_zero_mask] - y_pred[non_zero_mask])
                    / y_test[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    st.subheader("검증 성능 지표 (평균기온 3차 다항회귀)")
    st.write(f"**RMSE** : {rmse:,.2f}")
    st.write(f"**MAE**  : {mae:,.2f}")
    st.write(f"**R²**   : {r2:,.4f}")
    st.write(f"**MAPE** : {mape:,.2f} %")

    # 예측 vs 실제 그래프
    result_df = pd.DataFrame(
        {
            "날짜": test_dates.values,
            "실제": y_test,
            "예측": y_pred,
        }
    ).sort_values("날짜")

    st.subheader("검증 구간: 실제 vs 예측 (평균기온 3차 다항회귀)")
    st.line_chart(result_df.set_index("날짜")[["실제", "예측"]])

    # =========================
    # 6. 임의의 평균기온 입력 → 예측
    # =========================
    st.subheader("임의의 평균기온으로 공급량 예측하기")
    # 데이터 범위 기반으로 슬라이더 범위 설정
    temp_min = float(df[temp_col].min())
    temp_max = float(df[temp_col].max())

    temp_input = st.slider(
        "예측에 사용할 평균기온(°C)",
        min_value=float(np.floor(temp_min)),
        max_value=float(np.ceil(temp_max)),
        value=float(df[temp_col].iloc[-1]),
        step=0.5,
    )

    if st.button("이 평균기온으로 공급량 예측"):
        pred_supply = model.predict(np.array([[temp_input]]))[0]
        st.write(
            f"평균기온 **{temp_input:.1f} °C** 일 때 예측 공급량 ({target_col}): "
            f"**{pred_supply:,.2f}**"
        )

else:
    st.info("왼쪽 설정을 맞춘 뒤 **[실행 (학습 또는 불러오기)]** 버튼을 눌러줘.")
