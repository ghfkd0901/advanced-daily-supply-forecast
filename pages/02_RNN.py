import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

import plotly.graph_objects as go


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="일별 공급량 예측 (RNN, 시퀀스 기반)",
    layout="wide",
)

st.title("일별 공급량 예측 (RNN, 시퀀스 기반)")


# =========================
# 경로 & 상수
# =========================
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1lolxwf57MbcVTRWSR4sMTv7Z6TYelteb3BL33sPgFoo/export?format=csv&gid=0"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "daily_supply_rnn.keras")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_END_DATE = pd.Timestamp("2021-12-31")
TARGET_COL = "공급량(M3)"


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

required_cols = ["날짜", TARGET_COL, "평균기온"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()


# =========================
# 1. 사이드바 설정
# =========================
st.sidebar.header("모델 설정")

seq_len = st.sidebar.slider(
    "시퀀스 길이 (며칠을 보고 다음 날 예측?)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
)

rnn_type = st.sidebar.selectbox("RNN 타입", ["LSTM", "GRU"])

hidden_units_1 = st.sidebar.slider(
    "1층 RNN 유닛 수",
    min_value=16,
    max_value=256,
    value=128,
    step=16,
)

hidden_units_2 = st.sidebar.slider(
    "2층 RNN 유닛 수",
    min_value=8,
    max_value=128,
    value=64,
    step=8,
)

dropout_rate = st.sidebar.slider(
    "Dropout 비율",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05,
)

learning_rate = st.sidebar.number_input(
    "학습률 (learning rate)",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f",
)

epochs = st.sidebar.slider("학습 Epoch 수", 5, 200, 60, 5)
batch_size = st.sidebar.selectbox("배치 크기", [16, 32, 64, 128], index=1)

# --- 피처 선택 ---
st.sidebar.header("피처 선택")

# 이동평균 후보: 7, 14, 21, 30, 60, 90
all_rolling_windows = [7, 14, 21, 30, 60, 90]
default_rolling = [7, 30]
selected_rollings = st.sidebar.multiselect(
    "사용할 이동평균 (일)",
    all_rolling_windows,
    default=default_rolling,
)

use_weekday = st.sidebar.checkbox("요일 one-hot 사용", True)
use_basic_holiday = st.sidebar.checkbox("공휴일 플래그(is_holiday) 사용", True)
use_seollal_chuseok = st.sidebar.checkbox("설/추석 상세 플래그 사용", True)

# --- 검증 기간 설정 ---
st.sidebar.header("검증 기간 설정")
default_eval_start = pd.to_datetime("2022-01-01").date()
default_eval_end = pd.to_datetime("2023-12-31").date()

eval_start_date = st.sidebar.date_input("검증 시작일", default_eval_start)
eval_end_date = st.sidebar.date_input("검증 종료일", default_eval_end)

eval_start = pd.Timestamp(eval_start_date)
eval_end = pd.Timestamp(eval_end_date)

if eval_start >= eval_end:
    st.sidebar.error("검증 종료일은 시작일보다 나중이어야 합니다.")

# --- 저장된 모델 사용 여부 ---
saved_exists = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(FEATURE_SCALER_PATH)
    and os.path.exists(TARGET_SCALER_PATH)
    and os.path.exists(CONFIG_PATH)
)

mode_options = ["새로 학습하기"]
if saved_exists:
    mode_options.append("저장된 모델 불러오기")

mode = st.sidebar.radio("모드 선택", mode_options)
save_after_train = st.sidebar.checkbox("학습 후 모델 저장", value=True)

run_button = st.sidebar.button("실행 (학습 또는 불러오기)", type="primary")


# =========================
# 2. 기본 전처리
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


df_raw[TARGET_COL] = to_float_series(df_raw[TARGET_COL])

temp_cols_available = []
for col in ["평균기온", "최고기온", "최저기온"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        temp_cols_available.append(col)

if "요일" not in df_raw.columns:
    df_raw["요일"] = df_raw["날짜"].dt.day_name()
if "공휴일" not in df_raw.columns:
    df_raw["공휴일"] = ""


# =========================
# 3. 피처 엔지니어링
# =========================
df = df_raw.copy()

# 이동평균 전부 계산
for w in all_rolling_windows:
    df[f"rolling{w}"] = df[TARGET_COL].rolling(window=w, min_periods=1).mean()

df["month"] = df["날짜"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

weekday_dummies = (
    pd.get_dummies(df["요일"], prefix="weekday") if use_weekday else pd.DataFrame(index=df.index)
)

holiday_str = df["공휴일"].fillna("").astype(str)

df["is_seollal"] = holiday_str.str.contains("설날|설 연휴", regex=True)
df["is_chuseok"] = holiday_str.str.contains("추석", regex=True)

df["is_normal_holiday"] = (
    (holiday_str.str.strip() != "")
    & ~df["is_seollal"]
    & ~df["is_chuseok"]
)

big_holiday = df["is_seollal"] | df["is_chuseok"]
df["is_big_holiday"] = big_holiday.astype(int)
df["is_big_holiday_before1"] = big_holiday.shift(-1, fill_value=False).astype(int)
df["is_big_holiday_after1"] = big_holiday.shift(1, fill_value=False).astype(int)

df["is_holiday"] = (
    df["is_normal_holiday"] | df["is_big_holiday"]
).astype(int)

for col in ["is_seollal", "is_chuseok", "is_normal_holiday"]:
    df[col] = df[col].astype(int)

feature_cols_base = []

feature_cols_base += temp_cols_available

for w in selected_rollings:
    feature_cols_base.append(f"rolling{w}")

feature_cols_base += ["month_sin", "month_cos"]

if use_basic_holiday:
    feature_cols_base.append("is_holiday")

if use_seollal_chuseok:
    feature_cols_base += [
        "is_seollal",
        "is_chuseok",
        "is_big_holiday_before1",
        "is_big_holiday_after1",
    ]

feature_df = df[feature_cols_base].reset_index(drop=True)

if use_weekday:
    feature_df = pd.concat(
        [feature_df, weekday_dummies.reset_index(drop=True)],
        axis=1,
    )

final_feature_cols = list(feature_df.columns)
target_series = df[TARGET_COL]

data_all = pd.concat(
    [
        df[["날짜"]].reset_index(drop=True),
        feature_df,
        target_series.reset_index(drop=True),
    ],
    axis=1,
)
data_all = data_all.dropna().reset_index(drop=True)

use_dates = data_all["날짜"]
feature_values = data_all[final_feature_cols].values
target_values = data_all[TARGET_COL].values

st.subheader("모델에 사용할 데이터 샘플")
st.write(f"사용 행 수: {len(data_all)}, 피처 수: {len(final_feature_cols)}")
st.dataframe(data_all.head(10))


# =========================
# 4. 유틸 함수
# =========================
def make_sequences(feat, tgt, dates, window):
    X_list, y_list, date_list = [], [], []
    for i in range(len(feat) - window):
        X_list.append(feat[i : i + window])
        y_list.append(tgt[i + window])
        date_list.append(dates.iloc[i + window])
    return np.array(X_list), np.array(y_list), date_list


def build_rnn_model(
    input_timesteps: int,
    input_features: int,
    rnn_type: str,
    units1: int,
    units2: int,
    dropout_rate: float,
    learning_rate: float,
):
    model = Sequential()
    rnn_layer = LSTM if rnn_type == "LSTM" else GRU

    model.add(
        rnn_layer(
            units1,
            return_sequences=True,
            input_shape=(input_timesteps, input_features),
        )
    )
    model.add(Dropout(dropout_rate))

    model.add(rnn_layer(units2))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae"],
    )
    return model


def save_all(model, feature_scaler, target_scaler, config_dict):
    with open(FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(target_scaler, f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    model.save(MODEL_PATH)


def load_all():
    with open(FEATURE_SCALER_PATH, "rb") as f:
        f_scaler = pickle.load(f)
    with open(TARGET_SCALER_PATH, "rb") as f:
        t_scaler = pickle.load(f)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    m = load_model(MODEL_PATH)
    return m, f_scaler, t_scaler, config


# =========================
# 5. 실행
# =========================
if run_button:

    if eval_start <= TRAIN_END_DATE:
        st.warning(
            "⚠️ 검증 시작일은 훈련 종료일(2021-12-31) 이후로 설정하는 것이 좋습니다."
        )

    if mode == "저장된 모델 불러오기":
        if not saved_exists:
            st.error("저장된 모델이 없습니다. 먼저 학습해서 저장해 주세요.")
            st.stop()

        st.info("저장된 모델과 스케일러를 불러옵니다.")
        model, feature_scaler, target_scaler, config = load_all()

        if (
            config.get("seq_len") != seq_len
            or config.get("feature_cols") != final_feature_cols
        ):
            st.warning(
                "저장된 모델이 학습될 때의 피처 구성/시퀀스 길이와 "
                "현재 설정이 다릅니다. '새로 학습하기'로 다시 학습하는 걸 추천합니다."
            )

        try:
            features_scaled = feature_scaler.transform(feature_values)
        except Exception as e:
            st.error(
                "저장된 모델이 이전 피처 구성으로 학습되어 있습니다.\n"
                "지금과 다른 피처 구성이면 transform이 안 맞을 수 있어요.\n"
                "'새로 학습하기'로 다시 학습해 주세요.\n\n"
                f"에러: {e}"
            )
            st.stop()

        target_scaled = target_scaler.transform(target_values.reshape(-1, 1))

    else:
        st.info("새 모델을 학습합니다.")
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_scaled = feature_scaler.fit_transform(feature_values)
        target_scaled = target_scaler.fit_transform(target_values.reshape(-1, 1))

    X_all, y_all, y_dates = make_sequences(
        features_scaled, target_scaled, use_dates, seq_len
    )
    if len(X_all) == 0:
        st.error("시퀀스 길이가 데이터 길이보다 길어서 학습할 수 없습니다.")
        st.stop()

    y_dates_series = pd.Series(y_dates)

    train_idx = y_dates_series <= TRAIN_END_DATE
    test_idx = (y_dates_series >= eval_start) & (y_dates_series <= eval_end)

    if not train_idx.any():
        st.error("훈련 구간(2021년까지)에 해당하는 시퀀스가 없습니다.")
        st.stop()
    if not test_idx.any():
        st.error("선택한 검증 기간에 해당하는 시퀀스가 없습니다.")
        st.stop()

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    dates_test = y_dates_series[test_idx].tolist()

    n_features = X_all.shape[-1]
    st.write(
        f"훈련 샘플 수: {X_train.shape[0]}, "
        f"검증 샘플 수: {X_test.shape[0]}, "
        f"피처 수: {n_features}"
    )

    history = None
    if mode == "새로 학습하기":
        with st.spinner("모델 학습 중..."):
            model = build_rnn_model(
                input_timesteps=seq_len,
                input_features=n_features,
                rnn_type=rnn_type,
                units1=hidden_units_1,
                units2=hidden_units_2,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

            callback_es = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[callback_es],
                verbose=0,
            )

        st.success("학습 완료!")

        if history is not None:
            hist_df = pd.DataFrame(history.history)

            st.subheader("학습 그래프")

            col1, col2 = st.columns(2)

            with col1:
                if {"loss", "val_loss"}.issubset(hist_df.columns):
                    st.write("### Loss 변화")
                    st.line_chart(hist_df[["loss", "val_loss"]])

            with col2:
                if {"mae", "val_mae"}.issubset(hist_df.columns):
                    st.write("### MAE 변화")
                    st.line_chart(hist_df[["mae", "val_mae"]])

        if save_after_train:
            config = {
                "seq_len": seq_len,
                "feature_cols": final_feature_cols,
                "rnn_type": rnn_type,
                "hidden_units_1": hidden_units_1,
                "hidden_units_2": hidden_units_2,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "train_end_date": TRAIN_END_DATE.strftime("%Y-%m-%d"),
            }
            save_all(model, feature_scaler, target_scaler, config)
            st.success(f"모델 및 스케일러 저장 완료: {MODEL_DIR}/")

    else:
        st.success("저장된 모델 로드 완료! (학습은 다시 하지 않음)")

    # =========================
    # 6. 검증 성능 + Plotly 그래프
    # =========================
    y_test_pred_scaled = model.predict(X_test, verbose=0)
    y_test_true = target_scaler.inverse_transform(y_test).flatten()
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)
    r2 = r2_score(y_test_true, y_test_pred)

    non_zero_mask = y_test_true != 0
    if non_zero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_test_true[non_zero_mask] - y_test_pred[non_zero_mask])
                    / y_test_true[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    st.subheader(f"검증 성능 지표 ({eval_start.date()} ~ {eval_end.date()})")
    st.write(f"**RMSE** : {rmse:,.2f}")
    st.write(f"**MAE**  : {mae:,.2f}")
    st.write(f"**R²**   : {r2:,.4f}")
    st.write(f"**MAPE** : {mape:,.2f} %")

    result_df = pd.DataFrame(
        {"날짜": dates_test, "실제": y_test_true, "예측": y_test_pred}
    ).sort_values("날짜")

    result_df["실제_예측_비율"] = np.where(
        result_df["예측"] != 0,
        result_df["실제"] / result_df["예측"],
        np.nan,
    )

    customdata = np.stack(
        [
            result_df["실제"].values,
            result_df["예측"].values,
            result_df["실제_예측_비율"].values,
        ],
        axis=-1,
    )

    fig = go.Figure()

    # 🔹 실제: hover 끄기 (라인+마커만 표시)
    fig.add_trace(
        go.Scatter(
            x=result_df["날짜"],
            y=result_df["실제"],
            mode="lines+markers",
            name="실제",
            hoverinfo="skip",
        )
    )

    # 🔹 예측: hover에 날짜 + 실제 + 예측 + 비율 모두 표시
    fig.add_trace(
        go.Scatter(
            x=result_df["날짜"],
            y=result_df["예측"],
            mode="lines+markers",
            name="예측",
            customdata=customdata,
            hovertemplate=(
                "날짜: %{x|%Y-%m-%d}<br>"
                "실제: %{customdata[0]:,.2f}<br>"
                "예측: %{customdata[1]:,.2f}<br>"
                "실제 / 예측: %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"검증 구간: 실제 vs 예측 ({eval_start.date()} ~ {eval_end.date()})",
        xaxis_title="날짜",
        yaxis_title="공급량(M3)",
        hovermode="x",
    )

    st.subheader(f"검증 구간: 실제 vs 예측 ({eval_start.date()} ~ {eval_end.date()})")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("왼쪽 설정을 맞춘 뒤 **[실행 (학습 또는 불러오기)]** 버튼을 눌러줘.")
