import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

WINDOW = 60  # số ngày quá khứ dùng để dự đoán ngày kế tiếp

st.set_page_config(page_title="Dự báo cổ phiếu LSTM", page_icon="📈", layout="wide")
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", "VNM.VN")

c1, c2, c3 = st.columns(3)
start = c1.date_input("📅 Chọn ngày bắt đầu:", dt.date(2018, 1, 1))
end = c2.date_input("📅 Chọn ngày kết thúc:", dt.date.today())
n_days = c3.number_input("🔮 Số ngày muốn dự báo:", min_value=1, max_value=60, value=10)


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def make_xy(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i, 0])
        y.append(arr[i, 0])
    return np.array(X)[..., np.newaxis], np.array(y)


def build_model():
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


if st.button("Tải dữ liệu & Chạy dự báo", type="primary"):
    with st.spinner("Đang tải dữ liệu..."):
        df = load_data(ticker.strip(), start, end)

    if df.empty or len(df) < WINDOW + 100:
        st.error("Không đủ dữ liệu. Kiểm tra lại mã cổ phiếu hoặc mở rộng khoảng ngày.")
        st.stop()

    st.subheader(f"Dữ liệu {ticker.upper()}")
    st.dataframe(df.tail(10))

    close = df[["Close"]].values
    scaler = MinMaxScaler()
    split = int(len(close) * 0.8)

    # Chỉ fit scaler trên tập train để tránh rò rỉ dữ liệu
    scaler.fit(close[:split])
    scaled = scaler.transform(close)

    X_train, y_train = make_xy(scaled[:split], WINDOW)
    X_test, y_test = make_xy(scaled[split - WINDOW:], WINDOW)

    with st.spinner("Đang huấn luyện mô hình LSTM (có thể mất 1–2 phút)..."):
        model = build_model()
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Đánh giá trên tập test
    pred_test = scaler.inverse_transform(model.predict(X_test, verbose=0))
    real_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(real_test, pred_test))
    mape = mean_absolute_percentage_error(real_test, pred_test) * 100

    m1, m2 = st.columns(2)
    m1.metric("RMSE (tập test)", f"{rmse:,.2f}")
    m2.metric("MAPE (tập test)", f"{mape:.2f}%")

    # Dự báo n ngày tới (đệ quy)
    seq = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    future = []
    for _ in range(int(n_days)):
        nxt = model.predict(seq, verbose=0)[0, 0]
        future.append(nxt)
        seq = np.append(seq[:, 1:, :], [[[nxt]]], axis=1)
    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=int(n_days))

    # Biểu đồ
    fig, ax = plt.subplots(figsize=(12, 5))
    test_idx = df.index[split:]
    ax.plot(df.index[-300:], df["Close"].iloc[-300:], label="Giá thực tế")
    ax.plot(test_idx[-len(pred_test):], pred_test, label="Dự đoán (test)")
    ax.plot(future_dates, future, "r--o", label=f"Dự báo {n_days} ngày")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Bảng giá dự báo")
    st.dataframe(pd.DataFrame({"Ngày": future_dates.date, "Giá dự báo": future.round(2)}))
    st.caption("⚠️ Chỉ phục vụ học tập, không phải khuyến nghị đầu tư.")
