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
        날짜 += timedelta(days=3)            # 3일 간격으로 (호출 줄이기)
        time.sleep(0.1)
    return pd.DataFrame(rows)

raw = collect_boxoffice()
st.write("모은 행:", len(raw), "| 영화 수:", raw["영화명"].nunique())

import numpy as np
raw["개봉일"] = pd.to_datetime(raw["개봉일"])
raw["날짜"]  = pd.to_datetime(raw["날짜"])
raw["경과일"] = (raw["날짜"] - raw["개봉일"]).dt.days

초기 = raw[(raw["개봉일"].dt.year == 2023) & raw["경과일"].between(0, 10)]
첫기록 = 초기.sort_values("날짜").groupby("영화명").first()      # 개봉 초기 성적
최종 = raw.groupby("영화명")["누적관객"].max()                  # 최종 흥행

df = 첫기록.join(최종.rename("최종관객")).dropna(subset=["스크린수"])
st.write("영화 수:", len(df))
st.write(df.sort_values("최종관객", ascending=False)[["스크린수","최종관객"]].head(3))
