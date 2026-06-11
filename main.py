import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="2026 흥행 궤적", layout="wide")

BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"


# ════════════════════════════════════════════════
# 1) 2026년 박스오피스 수집
# ════════════════════════════════════════════════
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


# ════════════════════════════════════════════════
# 2) 영화별 곡선 + 메타 + 개봉초기 특성
# ════════════════════════════════════════════════
@st.cache_data
def build_curves(raw):
    raw = raw.copy()
    raw["개봉일"] = pd.to_datetime(raw["개봉일"], errors="coerce")
    raw["날짜"] = pd.to_datetime(raw["날짜"], errors="coerce")
    raw = raw.dropna(subset=["개봉일", "날짜"])
    raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days
    raw = raw[raw["경과일"].between(0, 120)]

    오늘 = pd.Timestamp(date.today())
    curves, meta, feat = {}, [], {}

    for 영화, g in raw.groupby("영화명"):
        g = g.sort_values("경과일")
        if g["경과일"].max() < 3:
            continue

        # 경과일 → 누적관객 곡선 (구멍은 보간, 누적은 줄지 않게)
        idx = range(int(g["경과일"].min()), int(g["경과일"].max()) + 1)
        s = g.set_index("경과일")["누적관객"].reindex(idx)
        s = s.interpolate().ffill().bfill().cummax()
        curves[영화] = s

        # 개봉 첫 주(경과일 7 이내) 첫 기록 = 초기 특성
        초기후보 = g[g["경과일"] <= 7]
        초기 = 초기후보.iloc[0] if len(초기후보) else g.iloc[0]
        feat[영화] = {
            "스크린수": int(초기["스크린수"]),
            "상영횟수": int(초기["상영횟수"]),
            "순위": int(초기["순위"]),
        }

        # 종영 판정: 최근 7일간 Top10 기록 없으면 내림
        마지막날 = g["날짜"].max()
        meta.append({
            "영화명": 영화,
            "최종관객": int(s.iloc[-1]),
            "마지막경과일": int(s.index[-1]),
            "마지막날": 마지막날,
            "종영": (오늘 - 마지막날).days >= 7,
        })

    return curves, pd.DataFrame(meta).set_index("영화명"), feat


# ════════════════════════════════════════════════
# 앱 본문
# ════════════════════════════════════════════════
st.title("🎬 2026 흥행 궤적")
st.caption("올해 박스오피스 데이터로, 영화들의 누적 관객 궤적을 비교하고 예측합니다.")

raw = collect_2026()
curves, meta, raw_feat = build_curves(raw)
st.write(f"분석 대상 영화: {len(curves)}편 (상영 종료 추정 {int(meta['종영'].sum())}편)")

top20 = meta.sort_values("최종관객", ascending=False).head(20)

tab1, tab2 = st.tabs(["📈 Top 20 궤적 비교", "🔮 흥행 예측 (A·B·C)"])

# ─────────────────────────────────────────────
# 탭 1 : Top 20 궤적 비교
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
# 탭 2 : 흥행 예측 (세 모델 A·B·C)
# ─────────────────────────────────────────────
with tab2:
    st.caption("세 가지 방식(A·B·C)으로 각각 최종 관객을 예측하고, 올해 몇 위 수준인지도 봅니다. "
               "세 모델 모두 '현재 누적'에서 출발해, 현재를 무시한 엉뚱한 예측을 막았어요.")

    대상 = st.selectbox("예측할 영화", top20.index.tolist())
    target = curves[대상]
    now_day = int(target.index[-1])
    now_acc = float(target.iloc[-1])

    cc1, cc2 = st.columns(2)
    cc1.metric("현재 경과일", f"개봉 {now_day}일째")
    cc2.metric("현재 누적 관객", f"{now_acc:,.0f} 명")

    # 이미 끝난 영화면 예측 대신 안내
    if meta.loc[대상, "종영"]:
        st.success(f"**'{대상}'은(는) 이미 상영을 마친 것으로 보여요.** "
                   f"최종 누적은 현재와 비슷한 **{now_acc:,.0f}명**으로 확정에 가깝습니다. "
                   "아직 상영 중인 영화를 고르면 A·B·C 예측을 볼 수 있어요.")
        st.stop()

    # ── 학습용: 끝난(최종값 신뢰되는) 과거 영화들 ──
    완성작 = [영화 for 영화 in curves
            if 영화 != 대상
            and meta.loc[영화, "종영"]
            and curves[영화].index[-1] >= now_day]

    if len(완성작) < 5:
        st.warning("아직 학습에 쓸 '끝난 과거 영화'가 부족해요 (5편 이상 필요). "
                   "데이터가 더 쌓이면 정확해집니다.")
        st.stop()

    예측결과 = {}

    # ── 모델 A : 비율 기반 ──
    비율들 = []
    for 영화 in 완성작:
        s = curves[영화]
        if now_day in s.index and s.iloc[-1] > 0:
            비율들.append(float(s.loc[now_day]) / float(s.iloc[-1]))
    if 비율들:
        중앙비율 = min(max(float(np.median(비율들)), 0.05), 1.0)
        예측결과["A · 비율 기반"] = now_acc / 중앙비율

    # ── 모델 B : ML 회귀 (랜덤포레스트) ──
    Xrows, yrows = [], []
    for 영화 in 완성작:
        s = curves[영화]
        주간 = raw_feat.get(영화)
        if 주간 is None or now_day not in s.index:
            continue
        Xrows.append([
            np.log1p(주간["스크린수"]), np.log1p(주간["상영횟수"]),
            주간["순위"], np.log1p(float(s.loc[now_day])), now_day,
        ])
        yrows.append(np.log1p(float(s.iloc[-1])))

    if len(Xrows) >= 5:
        rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(Xrows, yrows)
        내주간 = raw_feat[대상]
        x_me = [[np.log1p(내주간["스크린수"]), np.log1p(내주간["상영횟수"]),
                 내주간["순위"], np.log1p(now_acc), now_day]]
        pred_b = float(np.expm1(rf.predict(x_me)[0]))
        예측결과["B · ML 회귀"] = max(pred_b, now_acc)

    # ── 모델 C : 자기 곡선 외삽 ──
    daily = target.diff().dropna()
    if len(daily) >= 5:
        오늘관객 = float(daily.iloc[-1])
        감소율 = 0.93
        누적, 하루 = now_acc, max(오늘관객, 1.0)
        for _ in range(120):
            하루 *= 감소율
            누적 += 하루
            if 하루 < max(오늘관객, 1) * 0.02:
                break
        예측결과["C · 곡선 외삽"] = 누적

    if not 예측결과:
        st.warning("예측에 필요한 데이터가 부족해요. 데이터가 더 쌓이면 가능해집니다.")
        st.stop()

    # ── 순위 예측: 올해 끝난 영화들 최종값과 비교 ──
    올해최종 = meta[meta["종영"]]["최종관객"].sort_values(ascending=False).values
    def 예상순위(값):
        return int((올해최종 > 값).sum()) + 1

    # ── 결과 카드 ──
    st.subheader("📊 모델별 예상 최종 관객")
    색 = {"A · 비율 기반": "orange", "B · ML 회귀": "#4da6ff", "C · 곡선 외삽": "#2ecc71"}
    cols = st.columns(len(예측결과))
    for col, (이름, 값) in zip(cols, 예측결과.items()):
        col.metric(이름, f"{int(값):,} 명", f"올해 {예상순위(값)}위 수준")

    평균예측 = float(np.mean(list(예측결과.values())))
    st.info(f"**세 모델 평균 ≈ {int(평균예측):,}명** · 올해 **{예상순위(평균예측)}위** 수준으로 끝날 전망")

    # ── 그래프 ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines+markers",
        line=dict(width=4, color="crimson"), name=f"⭐ {대상} (현재까지)"))
    for 이름, 값 in 예측결과.items():
        fig2.add_trace(go.Scatter(
            x=[now_day, now_day + 60], y=[now_acc, 값], mode="lines+markers",
            line=dict(dash="dot", width=2.5, color=색.get(이름, "gray")),
            name=f"{이름} → {int(값):,}"))
    fig2.add_vline(x=now_day, line_dash="dash", line_color="gray", annotation_text="지금")
    fig2.update_layout(title=f"'{대상}'의 예상 최종 — 세 모델 비교",
        xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
        height=560, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("A는 과거 영화들의 '개봉 N일째 = 최종의 몇 %' 비율로, B는 개봉 초기 성적을 학습한 "
               "랜덤포레스트로, C는 이 영화 자신의 최근 추세를 이어붙여 예측합니다. "
               "세 값이 비슷하면 신뢰할 만하고, 크게 갈리면 그만큼 불확실하다는 신호예요.")
