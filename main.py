import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="영화 흥행 궤적 예측기", layout="wide")

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
    raw = raw[raw["경과일"].between(0, 120)]   # 개봉 후 0~120일만

    오늘 = pd.Timestamp(date.today())
    curves = {}      # 영화명 -> Series(index=경과일, value=누적관객)
    meta = []        # 영화별 요약 정보

    for 영화, g in raw.groupby("영화명"):
        g = g.sort_values("경과일")
        if g["경과일"].max() < 3:          # 너무 짧으면 제외
            continue

        # 경과일 0 ~ 마지막까지 빈칸 없이 채우고, 누적은 단조증가로 보간
        idx = range(int(g["경과일"].min()), int(g["경과일"].max()) + 1)
        s = g.set_index("경과일")["누적관객"].reindex(idx)
        s = s.interpolate().ffill().bfill()
        s = s.cummax()                     # 누적은 줄지 않도록 보정
        curves[영화] = s

        # '상영 종료' 추정: 마지막 기록이 14일 이상 전이면 끝난 영화로 봄
        마지막날 = g["날짜"].max()
        끝남 = (오늘 - 마지막날).days >= 14
        meta.append({
            "영화명": 영화,
            "최종관객": int(s.iloc[-1]),
            "마지막경과일": int(s.index[-1]),
            "끝남": 끝남,
            "마지막날": 마지막날,
        })

    return curves, pd.DataFrame(meta).set_index("영화명")


# 3) 비슷한 과거 영화 찾기 (현재 경과일의 누적관객 기준 = 1-A)
def find_similar(target_curve, curves, meta, k=5):
    now_day = int(target_curve.index[-1])          # 내 영화의 현재 경과일
    now_acc = float(target_curve.iloc[-1])         # 현재 누적관객

    후보 = []
    for 영화, s in curves.items():
        info = meta.loc[영화]
        # 끝난 영화 중, 내 현재 경과일 이후까지 충분히 상영한 것만
        if not info["끝남"]:
            continue
        if s.index[-1] < now_day + 5:              # 내 시점 이후 더 보여줄 게 있어야
            continue
        if now_day not in s.index:
            continue
        그시점누적 = float(s.loc[now_day])
        거리 = abs(그시점누적 - now_acc)           # 같은 경과일의 누적 차이
        후보.append((영화, 거리, 그시점누적))

    후보.sort(key=lambda x: x[1])
    return [영화 for 영화, _, _ in 후보[:k]]


# ── 앱 본문 ──
st.title("🎬 흥행 궤적 예측기")
st.caption("상영 중인 영화가 '이미 끝난 비슷한 영화'처럼 흘러간다면 어떻게 끝날지 그려 봅니다.")

raw = collect_2026()
curves, meta = build_curves(raw)
st.write(f"분석 대상 영화: {len(curves)}편 (끝난 영화 {int(meta['끝남'].sum())}편)")

# 아직 상영 중인 영화만 예측 대상으로
상영중 = meta[~meta["끝남"]].sort_values("최종관객", ascending=False)
if len(상영중) == 0:
    st.warning("아직 상영 중인 영화가 충분치 않습니다. 데이터가 더 쌓이면 예측할 수 있어요.")
    st.stop()

대상 = st.selectbox("예측할 영화 (상영 중)", 상영중.index.tolist())
k = st.slider("비교할 비슷한 영화 수", 3, 8, 5)

target_curve = curves[대상]
유사목록 = find_similar(target_curve, curves, meta, k=k)

now_day = int(target_curve.index[-1])
now_acc = int(target_curve.iloc[-1])

col1, col2 = st.columns(2)
col1.metric("현재 경과일", f"개봉 {now_day}일째")
col2.metric("현재 누적 관객", f"{now_acc:,} 명")

if not 유사목록:
    st.warning("아직 비슷한 '끝난 영화'를 찾기 어렵습니다. 경과일이 더 쌓이면 정확해져요.")
    st.stop()

# ── 그래프 (2-C: 내 영화 실선 + 비슷한 과거 점선 + 평균 예상선) ──
fig = go.Figure()

# 비슷한 과거 영화들의 전체 곡선 (점선)
예상끝값 = []
for 영화 in 유사목록:
    s = curves[영화]
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(dash="dot", width=1.3),
        name=f"{영화} (최종 {int(s.iloc[-1]):,})",
        opacity=0.6,
    ))
    예상끝값.append(int(s.iloc[-1]))

# 비슷한 영화들의 '평균 곡선' = 예상 궤적 (굵은 선)
공통길이 = min(curves[영화].index[-1] for 영화 in 유사목록)
평균 = []
for d in range(0, 공통길이 + 1):
    vals = [float(curves[영화].loc[d]) for 영화 in 유사목록 if d in curves[영화].index]
    평균.append(np.mean(vals))
fig.add_trace(go.Scatter(
    x=list(range(len(평균))), y=평균, mode="lines",
    line=dict(width=4, color="orange"),
    name="📈 예상 궤적 (비슷한 영화 평균)",
))

# 내 영화 (현재까지 실선, 굵게)
fig.add_trace(go.Scatter(
    x=target_curve.index, y=target_curve.values, mode="lines+markers",
    line=dict(width=4, color="crimson"),
    name=f"⭐ {대상} (현재까지)",
))
# '지금 여기' 표시
fig.add_vline(x=now_day, line_dash="dash", line_color="gray",
              annotation_text="지금")

fig.update_layout(
    title=f"'{대상}'은 앞으로 어떻게 끝날까?",
    xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
    height=550, hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# 예상 최종 관객 요약
st.subheader("예상 최종 관객 수")
c1, c2, c3 = st.columns(3)
c1.metric("낙관 (최대)", f"{max(예상끝값):,} 명")
c2.metric("예상 (평균)", f"{int(np.mean(예상끝값)):,} 명")
c3.metric("비관 (최소)", f"{min(예상끝값):,} 명")

st.caption(
    "현재 경과일 시점의 누적 관객이 비슷했던, 이미 끝난 영화들을 찾아 그 궤적을 빌려온 예측입니다. "
    "입소문·경쟁작 같은 변수는 담기지 않으며, 일별 top 10만 모은 데이터라 작은 영화는 빠져 있습니다(선택 편향)."
)
