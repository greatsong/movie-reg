import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="2026 흥행 궤적 비교", layout="wide")

BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"


# 1) 2026년 박스오피스 수집
@st.cache_data(show_spinner=False)
def collect_2026():
    KEY = st.secrets["KOBIS_KEY"]
    start = date(2026, 1, 1)
    end = date.today() - timedelta(days=1)
    총일수 = (end - start).days + 1

    진행바 = st.progress(0.0)
    상태 = st.empty()

    rows = []
    날짜, 처리 = start, 0
    while 날짜 <= end:
        try:
            r = requests.get(BASE,
                params={"key": KEY, "targetDt": 날짜.strftime("%Y%m%d")}, timeout=10)
            목록 = r.json().get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
            for m in 목록:
                rows.append({
                    "영화명": m["movieNm"], "개봉일": m["openDt"], "날짜": 날짜,
                    "관객수": int(m["audiCnt"]), "누적관객": int(m["audiAcc"]),
                    "스크린수": int(m["scrnCnt"]), "상영횟수": int(m["showCnt"]),
                    "순위": int(m["rank"]),
                })
        except Exception:
            pass
        처리 += 1
        진행바.progress(처리 / 총일수)
        상태.write(f"📥 수집 중… {날짜:%Y-%m-%d} ({처리}/{총일수}일) · 누적 {len(rows):,}행")
        날짜 += timedelta(days=1)
        time.sleep(0.05)

    진행바.empty(); 상태.empty()
    return pd.DataFrame(rows)


# 2) 영화별 '경과일 → 누적관객' 곡선 만들기 (구멍은 보간으로 채움)
@st.cache_data
def build_curves(raw):
    raw = raw.copy()
    raw["개봉일"] = pd.to_datetime(raw["개봉일"], errors="coerce")
    raw["날짜"] = pd.to_datetime(raw["날짜"], errors="coerce")
    raw = raw.dropna(subset=["개봉일", "날짜"])
    raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days
    raw = raw[raw["경과일"].between(0, 120)]

    curves = {}
    meta = []
    for 영화, g in raw.groupby("영화명"):
        g = g.sort_values("경과일")
        if g["경과일"].max() < 3:
            continue
        idx = range(int(g["경과일"].min()), int(g["경과일"].max()) + 1)
        s = g.set_index("경과일")["누적관객"].reindex(idx)
        s = s.interpolate().ffill().bfill().cummax()   # 누적은 줄지 않도록
        curves[영화] = s
        meta.append({"영화명": 영화, "최종관객": int(s.iloc[-1]),
                     "마지막경과일": int(s.index[-1])})

    return curves, pd.DataFrame(meta).set_index("영화명")


# ── 앱 본문 ──
st.title("🎬 2026 흥행 궤적 비교 — Top 20")
st.caption("올해 누적 관객 상위 20편의 '개봉 후 누적 곡선'을 한자리에서 겹쳐 봅니다.")

raw = collect_2026()
curves, meta = build_curves(raw)
st.write(f"분석 대상 영화: {len(curves)}편")

# 올해 누적 관객 Top 20
top20 = meta.sort_values("최종관객", ascending=False).head(20)

# ── 보기 옵션 ──
c1, c2 = st.columns([2, 1])
강조 = c1.multiselect("강조할 영화 (선택 안 하면 전체 동일)", top20.index.tolist())
로그축 = c2.toggle("세로축 로그 보기", value=False)

# ── 그래프: Top 20 곡선 겹쳐 그리기 ──
fig = go.Figure()
for 영화 in top20.index:
    s = curves[영화]
    굵게 = (영화 in 강조) or (len(강조) == 0)
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        name=f"{영화} ({int(s.iloc[-1]):,})",
        line=dict(width=3 if (영화 in 강조) else 1.8),
        opacity=1.0 if 굵게 else 0.25,
    ))

fig.update_layout(
    title="2026 Top 20 영화의 개봉 후 누적 관객 궤적",
    xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
    yaxis_type="log" if 로그축 else "linear",
    height=600, hovermode="x unified", legend=dict(font=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

# ── 표 ──
st.subheader("2026 누적 관객 Top 20")
표 = top20.reset_index()[["영화명", "최종관객", "마지막경과일"]].copy()
표.index = 표.index + 1
표.columns = ["영화명", "누적 관객", "집계된 마지막 경과일"]
표["누적 관객"] = 표["누적 관객"].map(lambda v: f"{v:,}")
st.dataframe(표, use_container_width=True)

st.caption(
    "각 곡선의 끝점이 현재까지의 누적 관객입니다. 위로 가파르게 솟은 영화일수록 "
    "초반에 빠르게 관객을 모은 작품이에요. 일별 top 10만 모은 데이터라 "
    "한때 11위 밖으로 밀려난 구간은 보간으로 메워 그렸습니다."
)
