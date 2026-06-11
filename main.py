import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="2026 흥행 궤적", layout="wide")

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


# 2) 영화별 '경과일 → 누적관객' 곡선 + 종영 여부
@st.cache_data
def build_curves(raw):
    raw = raw.copy()
    raw["개봉일"] = pd.to_datetime(raw["개봉일"], errors="coerce")
    raw["날짜"] = pd.to_datetime(raw["날짜"], errors="coerce")
    raw = raw.dropna(subset=["개봉일", "날짜"])
    raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days
    raw = raw[raw["경과일"].between(0, 120)]

    오늘 = pd.Timestamp(date.today())
    curves, meta = {}, []
    for 영화, g in raw.groupby("영화명"):
        g = g.sort_values("경과일")
        if g["경과일"].max() < 3:
            continue
        idx = range(int(g["경과일"].min()), int(g["경과일"].max()) + 1)
        s = g.set_index("경과일")["누적관객"].reindex(idx)
        s = s.interpolate().ffill().bfill().cummax()   # 누적은 줄지 않도록
        curves[영화] = s

        마지막날 = g["날짜"].max()
        meta.append({
            "영화명": 영화,
            "최종관객": int(s.iloc[-1]),
            "마지막경과일": int(s.index[-1]),
            "마지막날": 마지막날,
            "종영": (오늘 - 마지막날).days >= 7,   # 7일 넘게 Top10에 없으면 내림
        })

    return curves, pd.DataFrame(meta).set_index("영화명")


# ── 앱 본문 ──
st.title("🎬 2026 흥행 궤적")
st.caption("올해 박스오피스 데이터로, 영화들의 누적 관객 궤적을 비교하고 예측합니다.")

raw = collect_2026()
curves, meta = build_curves(raw)
st.write(f"분석 대상 영화: {len(curves)}편 (상영 종료 추정 {int(meta['종영'].sum())}편)")

top20 = meta.sort_values("최종관객", ascending=False).head(20)

tab1, tab2 = st.tabs(["📈 Top 20 궤적 비교", "🔮 흥행 예측"])

# ─────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([2, 1])
    강조 = c1.multiselect("강조할 영화 (선택 안 하면 전체 동일)", top20.index.tolist())
    로그축 = c2.toggle("세로축 로그 보기", value=False)

    fig = go.Figure()
    for 영화 in top20.index:
        s = curves[영화]
        진하게 = (영화 in 강조) or (len(강조) == 0)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name=f"{영화} ({int(s.iloc[-1]):,})",
            line=dict(width=3 if (영화 in 강조) else 1.8),
            opacity=1.0 if 진하게 else 0.25,
        ))
    fig.update_layout(
        title="2026 Top 20 영화의 개봉 후 누적 관객 궤적",
        xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
        yaxis_type="log" if 로그축 else "linear",
        height=600, hovermode="x unified", legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
with tab2:
    st.caption("Top 20 중 한 편을 골라, 같은 경과일에 '아직 상영 중이던' 비슷한 영화들의 "
               "이후 궤적을 빌려 '앞으로 어떻게 끝날지'를 그려 봅니다.")

    대상 = st.selectbox("예측할 영화", top20.index.tolist())
    k = st.slider("비교할 비슷한 영화 수", 3, 8, 5)

    target = curves[대상]
    now_day = int(target.index[-1])
    now_acc = float(target.iloc[-1])

    cc1, cc2 = st.columns(2)
    cc1.metric("현재 경과일", f"개봉 {now_day}일째")
    cc2.metric("현재 누적 관객", f"{now_acc:,.0f} 명")

    # 내 영화가 이미 내려갔으면 예측하지 않는다
    if meta.loc[대상, "종영"]:
        st.success(
            f"**'{대상}'은(는) 이미 상영을 마친 것으로 보여요.** "
            f"(최근 7일간 Top10 기록이 없어요.) "
            f"최종 누적은 **{now_acc:,.0f}명** 선에서 마감됐을 가능성이 큽니다. "
            "아직 상영 중인 영화를 고르면 '앞으로의 궤적'을 예측해 볼 수 있어요."
        )
        fig_done = go.Figure()
        fig_done.add_trace(go.Scatter(x=target.index, y=target.values,
            mode="lines+markers", line=dict(width=4, color="crimson"), name=대상))
        fig_done.update_layout(title=f"'{대상}' 누적 관객 (상영 종료)",
            xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수", height=420)
        st.plotly_chart(fig_done, use_container_width=True)
        st.stop()

    # 비교군: 내 시점(now_day) 이후로도 상영이 이어진 영화만 (빌려올 궤적이 실제 존재)
    후보 = []
    for 영화 in top20.index:
        if 영화 == 대상:
            continue
        s = curves[영화]
        if now_day not in s.index:
            continue
        if s.index[-1] < now_day + 5:
            continue
        후보.append((영화, abs(float(s.loc[now_day]) - now_acc)))
    후보.sort(key=lambda x: x[1])
    유사 = [영화 for 영화, _ in 후보[:k]]

    if not 유사:
        st.warning("같은 시점에 '아직 상영 중이던' 비슷한 영화를 찾지 못했어요. "
                   "데이터가 더 쌓이면 비교가 가능해져요.")
        st.stop()

    fig2 = go.Figure()
    예상끝 = []
    for 영화 in 유사:
        s = curves[영화]
        fig2.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
            line=dict(dash="dot", width=1.3), opacity=0.6,
            name=f"{영화} (최종 {int(s.iloc[-1]):,})"))
        예상끝.append(int(s.iloc[-1]))

    공통끝 = min(curves[영화].index[-1] for 영화 in 유사)
    평균x, 평균y = [], []
    for d in range(0, 공통끝 + 1):
        vals = [float(curves[영화].loc[d]) for 영화 in 유사 if d in curves[영화].index]
        if vals:
            평균x.append(d); 평균y.append(np.mean(vals))
    fig2.add_trace(go.Scatter(x=평균x, y=평균y, mode="lines",
        line=dict(width=4, color="orange"), name="📈 예상 궤적 (평균)"))

    fig2.add_trace(go.Scatter(x=target.index, y=target.values,
        mode="lines+markers", line=dict(width=4, color="crimson"),
        name=f"⭐ {대상} (현재까지)"))
    fig2.add_vline(x=now_day, line_dash="dash", line_color="gray", annotation_text="지금")
    fig2.update_layout(title=f"'{대상}'은 앞으로 어떻게 끝날까?",
        xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
        height=560, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("예상 최종 관객 수")
    e1, e2, e3 = st.columns(3)
    e1.metric("낙관 (최대)", f"{max(예상끝):,} 명")
    e2.metric("예상 (평균)", f"{int(np.mean(예상끝)):,} 명")
    e3.metric("비관 (최소)", f"{min(예상끝):,} 명")

    st.caption("아직 상영 중이면서 누적이 비슷했던 영화들의 이후 궤적을 빌려온 예측입니다. "
               "이미 끝물인 영화는 비교에서 제외해, 다 끝난 영화가 다시 솟구치는 엉뚱한 예측을 막았어요.")
