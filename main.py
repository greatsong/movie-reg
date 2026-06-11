import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="흥행 궤적 예측", layout="wide")

# 깃허브 raw 주소 (본인 저장소로 바꾸세요). 세 해를 모두 불러와 합칩니다.
DATA_URLS = [
    "kobis_2024.csv",
    "kobis_2025.csv",
    "kobis_2026.csv",
]


# ════════════════════════════════════════════════
# 1) 세 해 데이터 읽어 합치기
# ════════════════════════════════════════════════
@st.cache_data
def load_all():
    dfs = [pd.read_csv(u) for u in DATA_URLS]
    raw = pd.concat(dfs, ignore_index=True)
    raw["개봉일"] = pd.to_datetime(raw["개봉일"], errors="coerce")
    raw["날짜"] = pd.to_datetime(raw["날짜"], errors="coerce")
    raw = raw.dropna(subset=["개봉일", "날짜"])
    # 연말연초 영화는 두 파일에 겹쳐 들어오므로 (영화명, 날짜)로 중복 제거
    raw = raw.drop_duplicates(subset=["영화명", "날짜"])
    raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days
    raw = raw[raw["경과일"].between(0, 200)]
    return raw


# ════════════════════════════════════════════════
# 2) 영화별 곡선 + 메타 + 개봉초기 특성
# ════════════════════════════════════════════════
@st.cache_data
def build_curves(raw):
    오늘 = pd.Timestamp(date.today())
    curves, meta, feat = {}, [], {}

    for 영화, g in raw.groupby("영화명"):
        g = g.sort_values("경과일")
        if g["경과일"].max() < 3:
            continue

        idx = range(int(g["경과일"].min()), int(g["경과일"].max()) + 1)
        s = g.set_index("경과일")["누적관객"].reindex(idx)
        s = s.interpolate().ffill().bfill().cummax()
        curves[영화] = s

        초기후보 = g[g["경과일"] <= 7]
        초기 = 초기후보.iloc[0] if len(초기후보) else g.iloc[0]
        feat[영화] = {
            "스크린수": int(초기["스크린수"]),
            "상영횟수": int(초기["상영횟수"]),
            "순위": int(초기["순위"]),
        }

        마지막날 = g["날짜"].max()
        개봉연도 = int(g["개봉일"].iloc[0].year)
        meta.append({
            "영화명": 영화,
            "최종관객": int(s.iloc[-1]),
            "마지막경과일": int(s.index[-1]),
            "마지막날": 마지막날,
            "개봉연도": 개봉연도,
            "종영": (오늘 - 마지막날).days >= 7,
        })

    return curves, pd.DataFrame(meta).set_index("영화명"), feat


# 포화 성장 곡선: 누적(t) = M * (1 - exp(-k*t))
def 포화곡선(t, M, k):
    return M * (1 - np.exp(-k * t))


# ════════════════════════════════════════════════
# 앱 본문
# ════════════════════════════════════════════════
st.title("🎬 흥행 궤적 예측")
st.caption("2024~2026년 박스오피스를 모두 학습해, 지금 상영 중인 영화의 최종 흥행을 예측합니다.")

raw = load_all()
curves, meta, raw_feat = build_curves(raw)

끝난수 = int(meta["종영"].sum())
상영중 = meta[~meta["종영"]].sort_values("최종관객", ascending=False)
st.write(f"분석 대상: 총 {len(curves)}편 · 학습용(상영 종료) {끝난수}편 · 현재 상영 중 {len(상영중)}편")

tab1, tab2 = st.tabs(["📈 궤적 비교", "🔮 흥행 예측 (A·B·C)"])

# ─────────────────────────────────────────────
# 탭 1 : 궤적 비교 (올해 개봉작 위주)
# ─────────────────────────────────────────────
with tab1:
    올해 = meta[meta["개봉연도"] >= 2025].sort_values("최종관객", ascending=False).head(20)
    c1, c2 = st.columns([2, 1])
    강조 = c1.multiselect("강조할 영화 (선택 안 하면 전체 동일)", 올해.index.tolist())
    로그축 = c2.toggle("세로축 로그 보기", value=False)

    fig = go.Figure()
    for 영화 in 올해.index:
        s = curves[영화]
        진하게 = (영화 in 강조) or (len(강조) == 0)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name=f"{영화} ({int(s.iloc[-1]):,})",
            line=dict(width=3 if (영화 in 강조) else 1.8),
            opacity=1.0 if 진하게 else 0.25,
        ))
    fig.update_layout(
        title="최근 개봉작의 개봉 후 누적 관객 궤적",
        xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
        yaxis_type="log" if 로그축 else "linear",
        height=600, hovermode="x unified", legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# 탭 2 : 흥행 예측 (세 모델 A·B·C)
# ─────────────────────────────────────────────
with tab2:
    st.caption("세 가지 방식(A·B·C)으로 각각 최종 관객을 예측하고, 올해(2026) 몇 위 수준인지도 봅니다. "
               "세 모델 모두 '현재 누적'에서 출발합니다.")

    if len(상영중) == 0:
        st.warning("지금 상영 중인 영화가 없어요. (모두 종영 추정)")
        st.stop()

    대상 = st.selectbox("예측할 영화 (상영 중)", 상영중.index.tolist())
    target = curves[대상]
    now_day = int(target.index[-1])
    now_acc = float(target.iloc[-1])

    cc1, cc2 = st.columns(2)
    cc1.metric("현재 경과일", f"개봉 {now_day}일째")
    cc2.metric("현재 누적 관객", f"{now_acc:,.0f} 명")

    # 학습용: 끝난(최종값 신뢰되는) 과거 영화들 — 2024·2025·2026 전체
    완성작 = [영화 for 영화 in curves
            if 영화 != 대상
            and meta.loc[영화, "종영"]
            and curves[영화].index[-1] >= now_day]

    예측결과 = {}
    curveB = None

    # ── 모델 A : 비율 기반 (과거 영화들의 평균 패턴) ──
    if len(완성작) >= 5:
        비율들 = []
        for 영화 in 완성작:
            s = curves[영화]
            if now_day in s.index and s.iloc[-1] > 0:
                비율들.append(float(s.loc[now_day]) / float(s.iloc[-1]))
        if 비율들:
            중앙비율 = min(max(float(np.median(비율들)), 0.05), 1.0)
            예측결과["A · 비율 기반"] = now_acc / 중앙비율

    # ── 모델 B : 포화곡선 피팅 (시계열에 가장 충실) ──
    xs = target.index.values.astype(float)
    ys = target.values.astype(float)
    if len(xs) >= 5:
        try:
            popt, _ = curve_fit(
                포화곡선, xs, ys,
                p0=[now_acc * 1.5, 0.05],
                bounds=([now_acc, 0.001], [now_acc * 8, 1.0]),
                maxfev=5000,
            )
            예측결과["B · 포화곡선 피팅"] = max(float(popt[0]), now_acc)
            curveB = popt
        except Exception:
            pass

    # ── 모델 C : 단순 외삽 (최근 추세 이어붙이기) ──
    daily = target.diff().dropna()
    if len(daily) >= 5:
        오늘관객 = float(daily.iloc[-1])
        감소율 = 0.93
        누적, 하루 = now_acc, max(오늘관객, 1.0)
        for _ in range(150):
            하루 *= 감소율
            누적 += 하루
            if 하루 < max(오늘관객, 1) * 0.02:
                break
        예측결과["C · 단순 외삽"] = 누적

    if not 예측결과:
        st.warning("예측에 필요한 데이터가 부족해요.")
        st.stop()

    # ── 순위 예측: 올해(2026) 개봉작 최종값과 비교 ──
    올해최종 = meta[meta["개봉연도"] == 2026]["최종관객"].sort_values(ascending=False).values
    def 예상순위(값):
        return int((올해최종 > 값).sum()) + 1

    # ── 결과 카드 ──
    st.subheader("📊 모델별 예상 최종 관객")
    색 = {"A · 비율 기반": "orange", "B · 포화곡선 피팅": "#4da6ff", "C · 단순 외삽": "#2ecc71"}
    cols = st.columns(len(예측결과))
    for col, (이름, 값) in zip(cols, 예측결과.items()):
        col.metric(이름, f"{int(값):,} 명", f"2026 {예상순위(값)}위 수준")

    평균예측 = float(np.mean(list(예측결과.values())))
    st.info(f"**세 모델 평균 ≈ {int(평균예측):,}명** · 2026년 **{예상순위(평균예측)}위** 수준으로 끝날 전망")

    if "B · 포화곡선 피팅" in 예측결과:
        st.caption("💡 데이터가 적을 땐 곡선 모양을 수식으로 박은 **B(포화곡선)**가 가장 믿을 만해요. "
                   "A·C는 그 B가 합리적인지 견주는 비교 기준입니다.")

    # ── 그래프 ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines+markers",
        line=dict(width=4, color="crimson"), name=f"⭐ {대상} (현재까지)"))

    if curveB is not None:
        xline = np.arange(0, now_day + 90)
        yline = 포화곡선(xline, *curveB)
        fig2.add_trace(go.Scatter(x=xline, y=yline, mode="lines",
            line=dict(width=2.5, color="#4da6ff", dash="dot"),
            name=f"B · 포화곡선 → {int(curveB[0]):,}"))

    for 이름, 값 in 예측결과.items():
        if 이름.startswith("B"):
            continue
        fig2.add_trace(go.Scatter(
            x=[now_day, now_day + 60], y=[now_acc, 값], mode="lines+markers",
            line=dict(dash="dot", width=2, color=색.get(이름, "gray")),
            name=f"{이름} → {int(값):,}"))

    fig2.add_vline(x=now_day, line_dash="dash", line_color="gray", annotation_text="지금")
    fig2.update_layout(title=f"'{대상}'의 예상 최종 — 세 모델 비교",
        xaxis_title="개봉 후 경과일", yaxis_title="누적 관객 수",
        height=560, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("A는 과거 영화들의 '개봉 N일째 = 최종의 몇 %' 비율(2024~2026 전체 학습), "
               "B는 이 영화 곡선을 포화 성장 공식 M·(1−e^(−kt))에 맞춰, "
               "C는 최근 추세를 이어붙여 예측합니다.")
