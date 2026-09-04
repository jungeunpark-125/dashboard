"""외래관광객조사 대시보드 - 데이터/코드북 로딩 유틸리티"""
import os
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATHS = {
    "pre": os.path.join(BASE_DIR, "data", "pre_covid.csv"),
    "post": os.path.join(BASE_DIR, "data", "post_covid.csv"),
}
DATASET_LABELS = {
    "pre": "pre_covid (2019 + 2023–2025)",
    "post": "post_covid (2023–2025)",
}
CODEBOOK_PATH = os.path.join(BASE_DIR, "codebook", "1_codebook_항목정의서_비교.xlsx")
CODEVALUE_PATH = os.path.join(BASE_DIR, "codebook", "2_코드값_연도비교_작업물.xlsx")

# Q1(원문항)과 D_MOK(정제변수)은 동일 코드체계 사용 -> 라벨 조회 시 별칭 처리
LABEL_ALIAS = {"Q1": "D_MOK"}

# pre_covid.csv에서 2019 통합 과정 중 설명형으로 재명명된 활동 더미 변수(코드북 미수록분) 수동 보강
MANUAL_DESC = {
    "식도락관광": ("문8. 참여한 활동(식도락 관광) - 참여여부 더미", "코드형"),
    "쇼핑": ("문8. 참여한 활동(쇼핑) - 참여여부 더미", "코드형"),
    "자연경관감상": ("문8. 참여한 활동(자연경관 감상) - 참여여부 더미", "코드형"),
    "고궁역사유적지방문": ("문8. 참여한 활동(고궁/역사 유적지 방문) - 참여여부 더미", "코드형"),
    "전통문화체험": ("문8. 참여한 활동(전통문화체험) - 참여여부 더미", "코드형"),
    "박물관전시관방문": ("문8. 참여한 활동(박물관, 전시관 관람) - 참여여부 더미", "코드형"),
    "연수교육연구": ("문8. 참여한 활동(연수/교육/연구) - 참여여부 더미", "코드형"),
    "기타활동": ("문8. 참여한 활동(기타) - 참여여부 더미", "코드형"),
    "한류공연장및촬영지방문": ("문8. 참여한 활동(K-POP/한류스타 공연장 및 촬영지 방문) - 참여여부 더미", "코드형"),
    "유흥오락": ("문8. 참여한 활동(유흥/오락) - 참여여부 더미", "코드형"),
    "뷰티의료관광": ("문8. 참여한 활동(뷰티/의료 관광) - 참여여부 더미", "코드형"),
    "스포츠레저관람및참가": ("문8. 참여한 활동(스포츠/레포츠 관람 및 참가) - 참여여부 더미", "코드형"),
    "업무수행": ("문8. 참여한 활동(업무 수행) - 참여여부 더미", "코드형"),
}

# 변수 탐색 탭에서 우선 노출할 핵심 변수 (그 외 변수는 전체 목록에서 선택 가능)
KEY_CATEGORICAL = ["survey_year", "D_BUN", "D_NAT", "D_SEX", "D_AGE", "D_MOK", "Q1", "D_NUM", "D_GUB", "TYP"]
KEY_NUMERIC_PRE = ["총액1인TOT2", "M박HAP", "M일HAP", "식음료비_합계", "현지교통비_합계",
                    "문화오락_합계", "기타_합계", "한국여행사지불비_합계", "콘도민박숙박통합"]
KEY_NUMERIC_POST = ["총액1인TOT2", "총액1인TOT_개별국제교통비제외2", "M박HAP", "M일HAP", "MDAY전체TOT_RAW61",
                     "숙박비1인대체", "음식점1인대체", "식음료1인대체", "쇼핑비1인대체", "여행사1인대체",
                     "문화서1인대체", "오락및1인대체", "치료및1인대체", "미용서1인대체", "가이드1인대체",
                     "대여서1인대체", "유류비1인대체", "데이터1인대체", "기타비1인대체"]

# 지출항목 상관관계에 쓰는 주요 지출항목 (데이터셋별)
EXPENDITURE_ITEMS = {
    "pre": ["식음료비_합계", "현지교통비_합계", "문화오락_합계", "기타_합계", "한국여행사지불비_합계"],
    "post": ["숙박비1인대체", "음식점1인대체", "식음료1인대체", "쇼핑비1인대체",
             "여행사1인대체", "문화서1인대체", "오락및1인대체", "치료및1인대체", "미용서1인대체", "기타비1인대체"],
}

REGIONS = ["서울", "경기", "인천", "강원", "대전", "충북", "충남", "세종", "경북", "경남",
           "대구", "울산", "부산", "광주", "전북", "전남", "제주"]


@st.cache_data(show_spinner="데이터 불러오는 중...")
def load_csv(name: str) -> pd.DataFrame:
    path = DATASET_PATHS[name]
    if not os.path.exists(path):
        st.error(
            f"데이터 파일을 찾을 수 없습니다: `{os.path.relpath(path, BASE_DIR)}`\n\n"
            "원자료(CSV)는 저장소에 포함되어 있지 않습니다. `dashboard/data/` 폴더에 "
            "`pre_covid.csv`, `post_covid.csv`를 직접 넣어주세요 (README 참고)."
        )
        st.stop()
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_codebook():
    """변수명 -> (설명, 데이터형태) 매핑"""
    raw = pd.read_excel(CODEBOOK_PATH, sheet_name="통합_코드북")
    desc, dtype = {}, {}
    for _, row in raw.iterrows():
        v = row.get("변수명")
        if pd.isna(v):
            continue
        desc[v] = row.get("변수설명")
        dtype[v] = row.get("데이터형태")
    for k, (d, t) in MANUAL_DESC.items():
        desc[k] = d
        dtype[k] = t
    return desc, dtype


@st.cache_data(show_spinner=False)
def load_code_labels():
    """변수명 -> {코드: 라벨} 매핑 (2025 라벨 우선, 없으면 2024, 2023 순으로 대체)"""
    raw = pd.read_excel(CODEVALUE_PATH, sheet_name="코드값_상세_작업본")
    labels = {}
    for _, row in raw.iterrows():
        var = row.get("변수명")
        code = row.get("코드")
        if pd.isna(var) or pd.isna(code):
            continue
        label = row.get("라벨_2025")
        if pd.isna(label):
            label = row.get("라벨_2024")
        if pd.isna(label):
            label = row.get("라벨_2023")
        if pd.isna(label):
            continue
        try:
            code_f = float(code)
            code_key = int(code_f) if code_f.is_integer() else code_f
        except (ValueError, TypeError):
            code_key = code
        labels.setdefault(var, {})[code_key] = str(label).strip()
    return labels


@st.cache_data(show_spinner=False)
def load_rename_history() -> pd.DataFrame:
    """2023~2025 사이 변수명이 바뀐 이력 (코드북 워크북 '변수명 변경(개요)' 시트)."""
    return pd.read_excel(CODEBOOK_PATH, sheet_name="변수명 변경(개요)")


@st.cache_data(show_spinner="pre/post 공통 변수 값 체계 비교 중...")
def compare_common_vars() -> pd.DataFrame:
    """pre_covid / post_covid에 공통으로 존재하는 변수들의 실제 값(코드/범위)이 서로 같은지 데이터로 직접 비교."""
    pre = load_csv("pre")
    post = load_csv("post")
    common = sorted((set(pre.columns) & set(post.columns)) - {"pnid"})

    rows = []
    for c in common:
        s_pre, s_post = pre[c], post[c]
        dtype_pre, dtype_post = str(s_pre.dtype), str(s_post.dtype)
        vals_pre = set(s_pre.dropna().unique().tolist())
        vals_post = set(s_post.dropna().unique().tolist())
        is_cat = len(vals_pre) <= 30 and len(vals_post) <= 30

        row = {
            "변수명": c,
            "dtype_pre": dtype_pre,
            "dtype_post": dtype_post,
            "고유값수_pre": len(vals_pre),
            "고유값수_post": len(vals_post),
        }
        if is_cat:
            only_pre = sorted(vals_pre - vals_post)
            only_post = sorted(vals_post - vals_pre)
            row["범주형"] = True
            row["pre에만 있는 코드"] = ", ".join(map(str, only_pre)) if only_pre else ""
            row["post에만 있는 코드"] = ", ".join(map(str, only_post)) if only_post else ""
            row["코드 체계 다름"] = bool(only_pre or only_post)
            row["범위_pre"] = ""
            row["범위_post"] = ""
        else:
            row["범주형"] = False
            row["pre에만 있는 코드"] = ""
            row["post에만 있는 코드"] = ""
            try:
                row["범위_pre"] = f"{s_pre.min():.1f} ~ {s_pre.max():.1f}"
                row["범위_post"] = f"{s_post.min():.1f} ~ {s_post.max():.1f}"
            except (TypeError, ValueError):
                row["범위_pre"] = row["범위_post"] = ""
            row["코드 체계 다름"] = None
        row["dtype 다름"] = dtype_pre != dtype_post
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["변수명", "범주형", "dtype_pre", "dtype_post", "dtype 다름",
            "고유값수_pre", "고유값수_post", "코드 체계 다름",
            "pre에만 있는 코드", "post에만 있는 코드", "범위_pre", "범위_post"]
    return df[cols]


@st.cache_data(show_spinner="지역별 방문 통계 계산 중...")
def region_visit_summary(name: str) -> pd.DataFrame:
    """지역별 숙박(박TOT>0) 방문율/평균 숙박일수 (비가중·가중)."""
    df = load_csv(name)
    w = df["weight"]
    rows = []
    for r in REGIONS:
        col = f"{r}박TOT"
        if col not in df.columns:
            continue
        visited = df[col].fillna(0) > 0
        rows.append({
            "지역": r,
            "방문자수(비가중)": int(visited.sum()),
            "비가중 방문율(%)": round(visited.mean() * 100, 1),
            "가중 방문율(%)": round(np.average(visited.astype(float), weights=w) * 100, 1),
            "평균 숙박일수(방문자중)": round(df.loc[visited, col].mean(), 1) if visited.sum() else None,
        })
    return pd.DataFrame(rows)


def get_var_desc(var: str, desc_map: dict, dtype_map: dict):
    d = desc_map.get(var)
    t = dtype_map.get(var)
    if d is None and var in LABEL_ALIAS:
        d = desc_map.get(LABEL_ALIAS[var])
        t = dtype_map.get(LABEL_ALIAS[var])
    return d, t


def get_labels_for(var: str, label_map: dict):
    m = label_map.get(var)
    if not m and var in LABEL_ALIAS:
        m = label_map.get(LABEL_ALIAS[var])
    return m or {}


def is_categorical(series: pd.Series, max_unique: int = 30) -> bool:
    return series.dropna().nunique() <= max_unique


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_w = np.cumsum(weights) - 0.5 * weights
    cum_w /= np.sum(weights)
    return float(np.interp(q, cum_w, values))


def weighted_freq_table(df: pd.DataFrame, col: str, weight_col: str = "weight", label_map: dict = None) -> pd.DataFrame:
    n = len(df)
    tmp = df[[col, weight_col]].copy()
    if label_map:
        tmp["라벨"] = tmp[col].map(label_map).fillna(tmp[col].astype(str))
    else:
        tmp["라벨"] = tmp[col].astype(str)
    g = tmp.groupby("라벨", observed=True)
    out = pd.DataFrame({
        "표본수(비가중)": g.size(),
        "가중추정치": g[weight_col].sum(),
    })
    out["비가중비율(%)"] = (out["표본수(비가중)"] / n * 100).round(2)
    out["가중비율(%)"] = (out["가중추정치"] / out["가중추정치"].sum() * 100).round(2)
    return out.sort_values("표본수(비가중)", ascending=False)
