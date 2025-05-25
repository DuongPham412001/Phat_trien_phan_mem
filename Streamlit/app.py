import streamlit as st
import pandas as pd
import altair as alt
import os

# Thiết lập cấu hình Streamlit
@st.cache_data
def load_data():
    overall_path = os.path.expanduser("overall_metrics.csv")
    comments_path = os.path.expanduser("parsed_comments.csv")

    overall_df = pd.read_csv(overall_path)
    comments_df = pd.read_csv(comments_path)

    return overall_df, comments_df

# Load data
st.title("🎤 Dashboard theo dõi độ nổi tiếng nghệ sĩ")
overall_df, comments_df = load_data()

# Sidebar - chọn nghệ sĩ
artist_list = overall_df["artist"].unique().tolist()
selected_artist = st.sidebar.selectbox("Chọn nghệ sĩ", artist_list)

# Hiển thị bảng tóm tắt
st.subheader("📊 Tổng quan chỉ số của nghệ sĩ")
st.dataframe(overall_df[overall_df["artist"] == selected_artist].set_index("artist"))

# Biểu đồ cảm xúc tổng quan
st.subheader("🧠 Tỉ lệ cảm xúc (tổng thể)")
artist_data = overall_df[overall_df["artist"] == selected_artist].iloc[0]
emotion_df = pd.DataFrame({
    "Emotion": ["Tích cực", "Tiêu cực", "Trung tính"],
    "Tỉ lệ": [artist_data["positive_pct"], artist_data["negative_pct"], artist_data["neutral_pct"]]
})

chart = alt.Chart(emotion_df).mark_bar().encode(
    x=alt.X("Emotion", sort=["Tích cực", "Trung tính", "Tiêu cực"]),
    y="Tỉ lệ",
    color="Emotion"
).properties(width=500)
st.altair_chart(chart)

# Biểu đồ cảm xúc theo thời gian
st.subheader("📈 Biến động cảm xúc theo thời gian")
filtered_comments = comments_df[comments_df["artist"] == selected_artist].copy()
filtered_comments["date"] = pd.to_datetime(filtered_comments["date"])
sentiment_trend = filtered_comments.groupby(["date", "sentiment_label"]).size().unstack().fillna(0)
sentiment_trend = sentiment_trend.rolling(3).mean()  # Làm mượt nhẹ

line_chart = alt.Chart(sentiment_trend.reset_index().melt("date")).mark_line().encode(
    x="date:T",
    y="value:Q",
    color="sentiment_label:N"
).properties(width=700)
st.altair_chart(line_chart)

# Hiển thị một số bình luận mẫu
st.subheader("💬 Một số bình luận gần đây")
num_comments = st.slider("Số lượng bình luận hiển thị", 5, 30, 10)
sampled_comments = filtered_comments.sort_values("date", ascending=False).head(num_comments)
for _, row in sampled_comments.iterrows():
    st.markdown(f"**🗓 {row['date'].strftime('%Y-%m-%d')}** — `{row['sentiment_label']}`")
    st.write(f"👉 {row['comment']}")
    st.divider()
