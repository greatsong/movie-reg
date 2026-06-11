import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from datetime import date, timedelta
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")

BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"


# 1) API로 2026년 박스오피스 수집 (진행 상황 표시)
@st.cache_data(show_spinner=False)
def collect_2026():
    KEY = st.secrets["KOBIS_KEY"]

    start = date(2026, 1, 1)
    end = date.today() - timedelta(days=1)   # 진행 중인 해라 어제까지만
    총일수 = (end - start).days + 1

    # 진행 상황을 그릴 자리
    진행바 = st.progress(0.0)
    상태 = st.empty()

    rows = []
    날짜 = start
    처리 = 0
    while 날짜 <= end:
        try:
            r = requests.get(
                BASE,
                params={"key": KEY, "targetDt": 날짜.strftime("%Y%m%d")},
                timeout=10,
            )
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
        상태.write(
            f"📥 수집 중… {날짜:%Y-%m-%d} "
            f"({처리}/{총일수}일, {처리/총일수*100:.0f}%) · 누적 {len(rows):,}행"
        )

        날짜 += timedelta(days=1)
        time.sleep(0.05)

    # 다 끝나면 진행 표시 지우기
    진행바.empty()
    상태.empty()
    return pd.DataFrame(rows)


# 2) 영화별 '개봉 초기 성적' + '최종 흥행'으로 정리
@st.cache_data
def build_movie_table(raw):
    raw = raw.copy()
    raw["개봉일"] = pd.to_datetime(raw["개봉일"], errors="coerce")
    raw["날짜"] = pd.to_datetime(raw["날짜"], errors="coerce")
    raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days

    초기 = raw[(raw["개봉일"].dt.year == 2026) & raw["경과일"].between(0, 10)]
    첫기록 = 초기.sort_values("날짜").groupby("영화명").first()
    최종 = raw.groupby("영화명")["누적관객"].max()

    df = 첫기록.join(최종.rename("최종관객")).dropna(subset=["스크린수"])
    df = df[df["최종관객"] > 0]

    # 새 단서(특성 공학): 한 상영당 평균 관객 = '열기'
    df["상영당관객"] = df["관객수"] / df["상영횟수"]
    return df


# 3) 모델 학습 (한 번만)
@st.cache_resource
def train_models(df):
    X = df[["스크린수", "상영횟수", "순위", "상영당관객"]].copy()
    X["스크린수"] = np.log1p(X["스크린수"])
    X["상영횟수"] = np.log1p(X["상영횟수"])
    X["상영당관객"] = np.log1p(X["상영당관객"])
    y = np.log1p(df["최종관객"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    후보 = {
        "선형회귀": LinearRegression(),
        "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=0),
        "그래디언트부스팅": GradientBoostingRegressor(random_state=0),
    }
    점수 = {}
    for 이름, m in 후보.items():
        m.fit(X_tr, y_tr)
        점수[이름] = round(r2_score(y_te, m.predict(X_te)), 3)

    # 배포용 최종 모델: 가장 단순한 선형회귀를 전체 데이터로 재학습
    최종모델 = LinearRegression().fit(X, y)

    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
    중요도 = pd.DataFrame(
        {"특성": ["스크린수", "상영횟수", "순위", "상영당관객"],
         "중요도": rf.feature_importances_}
    ).sort_values("중요도")

    return 최종모델, 점수, 중요도, list(X.columns)


# ── 앱 본문 ──
st.title("🎬 영화 흥행 예측기 — 2026 박스오피스")
st.caption("KOBIS API로 직접 모은 2026년 데이터로, 개봉 초기 성적만 보고 최종 관객 수를 예측합니다.")

raw = collect_2026()
st.write(f"수집한 기록: {len(raw):,}행 · 영화 {raw['영화명'].nunique() if len(raw) else 0}편")

df = build_movie_table(raw)

if len(df) < 10:
    st.warning(
        f"정리된 영화가 {len(df)}편뿐입니다. 2026년은 아직 진행 중이라 "
        "데이터가 적습니다. 시간이 지나 개봉작이 쌓이면 예측이 더 안정적이에요. "
        "(앱 우측 상단 ⋮ → Clear cache 로 최신 데이터를 다시 수집할 수 있습니다.)"
    )
    st.stop()

최종모델, 점수, 중요도, 컬럼 = train_models(df)

tab1, tab2, tab3 = st.tabs(["📊 데이터 살펴보기", "🤖 모델 성능", "🔮 흥행 예측기"])

with tab1:
    st.subheader(f"2026년 개봉작 {len(df)}편 (관객 순)")
    st.dataframe(
        df.sort_values("최종관객", ascending=False)[
            ["스크린수", "상영횟수", "순위", "상영당관객", "최종관객"]
        ].head(10)
    )

    df_plot = df.copy()
    df_plot["log스크린"] = np.log10(df_plot["스크린수"])
    df_plot["log최종관객"] = np.log10(df_plot["최종관객"])
    fig = px.scatter(
        df_plot, x="log스크린", y="log최종관객", hover_name=df_plot.index,
        title="개봉 초기 스크린 수 vs 최종 관객 수 (로그)",
        labels={"log스크린": "스크린 수(log)", "log최종관객": "최종 관객 수(log)"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("세 모델 견주기 (R², 1에 가까울수록 좋음)")
    점수표 = pd.DataFrame(점수.items(), columns=["모델", "R²"])
    st.dataframe(점수표, hide_index=True)
    st.caption("별 데이터처럼, 가장 단순한 선형회귀가 복잡한 모델 못지않은 경우가 많습니다.")

    st.subheader("무엇이 흥행을 좌우했나 (특성 중요도)")
    fig2 = px.bar(중요도, x="중요도", y="특성", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)
    st.info("특성 중요도는 '무엇을 단서로 삼았나'를 보여 줄 뿐, '원인'의 증거는 아닙니다. (상관 ≠ 인과)")

with tab3:
    st.subheader("개봉 초기 성적을 넣어 보세요")
    스크린 = st.slider("개봉 초기 스크린 수", 1, 2500, 800)
    상영 = st.slider("개봉 초기 상영 횟수", 1, 8000, 2000)
    순위 = st.slider("개봉 초기 순위", 1, 10, 3)
    열기 = st.slider("상영당 관객 수 (좌석 열기)", 1, 80, 20)

    입력 = pd.DataFrame(
        [[np.log1p(스크린), np.log1p(상영), 순위, np.log1p(열기)]],
        columns=컬럼,
    )
    예측 = np.expm1(최종모델.predict(입력)[0])   # 로그를 원래 관객 수로 되돌리기
    st.metric("예상 최종 관객 수", f"{int(예측):,} 명")

    st.caption(
        "참고용 예측입니다. 입소문·경쟁작 같은 '사람의 마음'은 숫자에 담기지 않고, "
        "이 데이터는 일별 top 10만 모은 것이라 작은 영화는 빠져 있습니다(선택 편향)."
    )
