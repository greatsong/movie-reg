import requests, time, pandas as pd
import streamlit as st
from datetime import date, timedelta

@st.cache_data
def collect_boxoffice():
    KEY = st.secrets["KOBIS_KEY"]          # 4장에서 배운 안전한 키 보관
    BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"

    rows = []
    날짜 = date(2023, 1, 1)
    while 날짜 <= date(2023, 12, 31):
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
