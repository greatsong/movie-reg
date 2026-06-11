import requests, time, pandas as pd
import streamlit as st
from datetime import date, timedelta

@st.cache_data
def collect_boxoffice():
    KEY = st.secrets["KOBIS_KEY"]          # 4장에서 배운 안전한 키 보관
    BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"

    rows = []
    날짜 = date(2026, 1, 1)
    while 날짜 <= date(2026, 6, 10):
        r = requests.get(BASE, params={"key": KEY, "targetDt": 날짜.strftime("%Y%m%d")})
        for m in r.json()["boxOfficeResult"]["dailyBoxOfficeList"]:
            rows.append({"영화명": m["movieNm"], "개봉일": m["openDt"], "날짜": 날짜,
                         "관객수": int(m["audiCnt"]), "누적관객": int(m["audiAcc"]),
                         "스크린수": int(m["scrnCnt"]), "상영횟수": int(m["showCnt"]),
                         "순위": int(m["rank"])})
        날짜 += timedelta(days=1)            # 3일 간격으로 (호출 줄이기)
        time.sleep(0.1)
    return pd.DataFrame(rows)

raw = collect_boxoffice()
st.write("모은 행:", len(raw), "| 영화 수:", raw["영화명"].nunique())

import numpy as np
raw["개봉일"] = pd.to_datetime(raw["개봉일"])
raw["날짜"]  = pd.to_datetime(raw["날짜"])
raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days

초기 = raw[(raw["개봉일"].dt.year == 2026) & raw["경과일"].between(0, 10)]
첫기록 = 초기.sort_values("날짜").groupby("영화명").first()      # 개봉 초기 성적
최종 = raw.groupby("영화명")["누적관객"].max()                  # 최종 흥행

df = 첫기록.join(최종.rename("최종관객")).dropna(subset=["스크린수"])
st.write("영화 수:", len(df))
st.write(df.sort_values("최종관객", ascending=False)[["스크린수","최종관객"]].head(3))
import plotly.express as px
df["log스크린"] = np.log10(df["스크린수"])
df["log최종관객"] = np.log10(df["최종관객"])
fig = px.scatter(df, x="log스크린", y="log최종관객", hover_name=df.index,
                 title="개봉 초기 스크린 수 vs 최종 관객 수 (로그)",
                 labels={"log스크린":"스크린 수(log)","log최종관객":"최종 관객 수(log)"})
st.plotly_chart(fig)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

X = df[["스크린수","상영횟수","순위"]].copy()
X["스크린수"] = np.log1p(X["스크린수"]); X["상영횟수"] = np.log1p(X["상영횟수"])
y = np.log1p(df["최종관객"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

for 이름, m in {"선형회귀": LinearRegression(),
               "랜덤포레스트": RandomForestRegressor(n_estimators=300, random_state=0),
               "그래디언트부스팅": GradientBoostingRegressor(random_state=0)}.items():
    m.fit(X_tr, y_tr)
    st.write(이름, round(r2_score(y_te, m.predict(X_te)), 3))
rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
imp = pd.DataFrame({"특성": X.columns, "중요도": rf.feature_importances_}).sort_values("중요도")
fig = px.bar(imp, x="중요도", y="특성", orientation="h", title="무엇이 흥행을 좌우했나")
st.plotly_chart(fig)
import streamlit as st

st.title("영화 흥행 예측기 — 개봉 초기 성적으로")
스크린 = st.slider("개봉 초기 스크린 수", 1, 2500, 800)
상영 = st.slider("개봉 초기 상영 횟수", 1, 8000, 2000)
순위 = st.slider("개봉 초기 순위", 1, 10, 3)
열기 = st.slider("상영당 관객 수(좌석 열기)", 1, 80, 20)

입력 = pd.DataFrame([[np.log1p(스크린), np.log1p(상영), 순위, np.log1p(열기)]],
                   columns=X2.columns)
예측 = np.expm1(개선모델.predict(입력)[0])      # 로그를 원래 관객 수로 되돌리기
st.metric("예상 최종 관객 수", f"{int(예측):,} 명")
