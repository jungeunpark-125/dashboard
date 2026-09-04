import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data_utils as du

st.set_page_config(page_title="외래관광객조사 대시보드", layout="wide")

DESC_MAP, DTYPE_MAP = du.load_codebook()
LABEL_MAP = du.load_code_labels()

st.title("방한 외국인 관광객 실태조사 대시보드")

tab1, tab2, tab3 = st.tabs(["데이터 원본", "PreCovid / PostCovid 비교", "변수 탐색"])

# ============================================================
# TAB 1. 데이터 원본
# ============================================================
with tab1:
    ds_name = st.radio("데이터셋 선택", options=["pre", "post"],
                        format_func=lambda x: du.DATASET_LABELS[x], horizontal=True, key="t1_ds")
    df = du.load_csv(ds_name)
    st.caption(f"{du.DATASET_LABELS[ds_name]} — {len(df):,}행 × {df.shape[1]}개 변수")

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        years = sorted(df["survey_year"].dropna().unique().tolist())
        sel_years = st.multiselect("연도 필터", years, default=years)
    with col_f2:
        nat_labels = du.get_labels_for("D_NAT", LABEL_MAP)
        nat_options = sorted(nat_labels.values()) if nat_labels else []
        sel_nat_labels = st.multiselect("국적 필터 (미선택 시 전체)", nat_options)
    with col_f3:
        default_cols = [c for c in du.KEY_CATEGORICAL + (du.KEY_NUMERIC_PRE if ds_name == "pre" else du.KEY_NUMERIC_POST) if c in df.columns]
        sel_cols = st.multiselect("표시할 변수", options=list(df.columns), default=default_cols[:12])

    view = df[df["survey_year"].isin(sel_years)] if sel_years else df
    if sel_nat_labels and nat_labels:
        inv_nat = {v: k for k, v in nat_labels.items()}
        codes = [inv_nat[v] for v in sel_nat_labels if v in inv_nat]
        view = view[view["D_NAT"].isin(codes)]

    n_show = st.slider("표시 행 수", 50, 5000, 500, step=50)
    st.dataframe(view[sel_cols].head(n_show) if sel_cols else view.head(n_show), use_container_width=True, height=420)
    st.caption(f"필터 적용 결과: {len(view):,}행 (표시는 상위 {min(n_show, len(view)):,}행)")

    st.download_button(
        "필터링된 전체 데이터 CSV 다운로드",
        data=(view[sel_cols] if sel_cols else view).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{ds_name}_filtered.csv", mime="text/csv",
    )

# ============================================================
# TAB 2. PreCovid / PostCovid 비교
# ============================================================
with tab2:
    df_pre = du.load_csv("pre")
    df_post = du.load_csv("post")
    pre_cols, post_cols = set(df_pre.columns), set(df_post.columns)
    common = pre_cols & post_cols
    pre_only = pre_cols - post_cols
    post_only = post_cols - pre_cols

    c1, c2, c3 = st.columns(3)
    c1.metric("공통 변수", len(common))
    c2.metric("pre_covid에만 존재", len(pre_only))
    c3.metric("post_covid에만 존재", len(post_only))

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.expander(f"pre_covid 전용 변수 목록 ({len(pre_only)}개)"):
            st.write(sorted(pre_only))
    with cc2:
        with st.expander(f"post_covid 전용 변수 목록 ({len(post_only)}개)"):
            st.write(sorted(post_only))

    with st.expander("pre_covid 집계변수는 이렇게 만들어졌습니다 — 펼쳐서 보기"):
        st.markdown(
            "2019년 원자료와 2023\\~2025년 통합자료는 일부 문항의 질문 형태와 세분화 수준이 달라 그대로 이어붙일 수 없었습니다. "
            "특히 2019는 응답한 항목을 나열하는 방식(gather)으로 저장돼 있어, 통합 데이터처럼 항목마다 열을 따로 두는 방식(spread)으로 "
            "다시 정리하는 매칭 작업에 품이 많이 들었습니다."
        )
        st.markdown("**pre_covid의 지출/숙박 집계변수는 아래처럼 2019 문항 + 통합 문항 여러 개를 합해서 만들었습니다.**")
        formula_tbl = pd.DataFrame([
            {"집계변수": "콘도민박숙박통합", "구성": "2019의 콘도·리조트·펜션 + 통합의 콘도미니엄/리조트 + 통합의 민박/펜션"},
            {"집계변수": "현지교통비_합계", "구성": "2019의 Q14_2_5 + 통합의 한국한국1인대체 + 한국국외1인대체 + 한국수상1인대체 + 한국철도1인대체 + 한국도로1인대체 + 대여서1인대체 + 유류비1인대체"},
            {"집계변수": "식음료비_합계", "구성": "2019의 Q14_2_4 + 통합의 음식점1인대체 + 식음료1인대체"},
            {"집계변수": "문화오락_합계", "구성": "2019의 Q14_2_7 + 통합의 문화서1인대체 + 오락및1인대체"},
            {"집계변수": "한국여행사지불비_합계", "구성": "2019의 Q14_2_6 + 통합의 여행사1인대체 + 가이드1인대체 + 단기투어상품1인대체"},
        ])
        st.dataframe(formula_tbl, use_container_width=True, hide_index=True)
        st.markdown(
            "**참고**: `한국여행사지불비_합계`는 통합 데이터 쪽 정의 범위가 가이드비·단기투어상품비까지 포함해 더 넓습니다. "
            "그래서 0원이 아닌 응답자 비율이 2019는 3.2%, 2023\\~2025는 22.6%로 차이가 큽니다 — 연도 비교 시 참고해주세요."
        )

    st.markdown("---")
    st.subheader("공통 변수의 값/코드 체계가 실제로 같은가?")
    st.caption("이름은 같지만 코드 정의가 다르면 pre/post를 그대로 이어붙여 분석할 때 오류가 생깁니다. 두 파일을 직접 읽어 비교했습니다.")

    cmp_df = du.compare_common_vars()
    n_dtype_diff = int(cmp_df["dtype 다름"].sum())
    n_code_diff = int((cmp_df["코드 체계 다름"] == True).sum())  # noqa: E712

    m1, m2, m3 = st.columns(3)
    m1.metric("비교한 공통 변수", len(cmp_df))
    m2.metric("dtype이 다른 변수", n_dtype_diff)
    m3.metric("코드값 체계가 다른 범주형 변수", n_code_diff)

    only_diff = st.checkbox("차이 나는 변수만 보기", value=True)
    show_df = cmp_df
    if only_diff:
        show_df = cmp_df[(cmp_df["dtype 다름"]) | (cmp_df["코드 체계 다름"] == True)]  # noqa: E712
    st.dataframe(show_df, use_container_width=True, height=420)
    st.caption(
        "'코드 체계 다름'은 범주형 변수(고유값 30개 이하)에서 한쪽 파일에만 등장하는 코드가 있다는 뜻입니다. "
        "범주형이 아닌 연속형 변수는 코드 비교 대신 값 범위(범위_pre/post)로 비교했습니다."
    )

    with st.expander("연도별 변수명 변경 이력"):
        st.dataframe(du.load_rename_history(), use_container_width=True, height=280)

# ============================================================
# TAB 3. 변수 탐색
# ============================================================
with tab3:
    ds_name3 = st.radio("데이터셋 선택", options=["pre", "post"],
                         format_func=lambda x: du.DATASET_LABELS[x], horizontal=True, key="t3_ds")
    df3 = du.load_csv(ds_name3)

    with st.expander("가중치(weight) 는 이렇게 처리했습니다 — 펼쳐서 보기"):
        st.markdown(
            "- `weight1`\\~`weight4` = **분기별 가중치**이지만 **모든 연도에 있는 게 아닙니다.** 실제로 값이 존재하는 건 "
            "**2024·2025년뿐**이고, 그 안에서도 응답자 1명당 자신이 응답한 분기 컬럼에만 값이 있고 나머지 3개는 결측입니다. "
            "**2019·2023년은 weight1\\~4 컬럼 자체가 전부 결측**입니다 (그 해에는 분기별 가중치를 별도 산출하지 않은 것으로 보입니다).\n"
            "- `weight` = **연간 통합 가중치**로, **2019\\~2025 전 연도에 걸쳐 결측이 0건**입니다. "
            "이 대시보드의 모든 '가중' 통계는 연도와 무관하게 **이 `weight` 컬럼만 사용**하기 때문에, 위의 연도별 weight1\\~4 결측 여부와 무관하게 "
            "모든 연도의 가중 계산이 정상적으로 이뤄집니다 (weight1\\~4는 분기 단위 세부분석이 필요할 때만 쓰는 보조 변수이며, 애초에 전 연도에 존재하지 않아 "
            "이 대시보드에서는 사용하지 않습니다).\n"
            "- **계산식**: 가중평균 = Σ(x·weight) / Σweight, 가중비율(%) = (카테고리별 weight 합) / (전체 weight 합) × 100.\n"
            "- **비가중(표본) vs 가중(모집단추정) 이 왜 다른가**: 조사 표본설계상 국적별로 표본을 고르게 뽑지 않고(소규모 국가를 상대적으로 더 뽑는 식) "
            "실제 방한객 규모에 맞춰 사후 보정한 게 `weight`입니다. 그래서 중국·일본처럼 실제 비중이 큰 국적은 표본비율보다 가중비율이 훨씬 높게 나옵니다 "
            "— 즉 **모집단(실제 방한 외국인 전체) 추정치로 해석하려면 항상 가중값을 봐야 합니다.**\n"
            "- 참고로 `weight`의 **단위(스케일)도 연도별로 다릅니다** — 2019년은 평균 1 안팎(비율형 보정계수)인 반면 2023\\~2025년은 평균 수백\\~천 단위(응답자 1명이 대표하는 인원수 규모)입니다. "
            "다만 가중평균·가중비율은 항상 그 연도 자체의 weight 합으로 나누는 비율 계산이라 **연도별 결과 자체는 스케일 차이의 영향을 받지 않습니다.**"
        )

    key_numeric = du.KEY_NUMERIC_PRE if ds_name3 == "pre" else du.KEY_NUMERIC_POST
    key_cat = list(du.KEY_CATEGORICAL)
    exclude = {"pnid", "weight", "weight1", "weight2", "weight3", "weight4"}
    # Q1과 D_MOK는 동일 변수의 중복(설문 원문항명 vs 표준화된 설계변수명) -> D_MOK가 있으면 Q1은 목록에서 숨김
    if "D_MOK" in df3.columns and "Q1" in df3.columns:
        key_cat = [c for c in key_cat if c != "Q1"]
        exclude.add("Q1")
    priority = [c for c in key_cat + key_numeric if c in df3.columns]
    other_cols = [c for c in df3.columns if c not in priority and c not in exclude]
    all_ordered = priority + sorted(other_cols)

    st.subheader("1) 변수별 분포 탐색")
    var = st.selectbox("변수 선택", all_ordered, index=0)

    desc, dtype = du.get_var_desc(var, DESC_MAP, DTYPE_MAP)
    label_map = du.get_labels_for(var, LABEL_MAP)
    desc_safe = (desc or "설명 정보 없음").replace("~", "\\~")
    st.info(f"**{var}** · 데이터형태: {dtype or '미상'}\n\n{desc_safe}")

    series = df3[var]
    categorical = du.is_categorical(series)

    if categorical:
        ft = du.weighted_freq_table(df3, var, label_map=label_map)
        ft = ft.head(25)
        fig = go.Figure()
        fig.add_bar(name="비가중(표본) %", x=ft.index, y=ft["비가중비율(%)"], marker_color="#2a78d6")
        fig.add_bar(name="가중(모집단추정) %", x=ft.index, y=ft["가중비율(%)"], marker_color="#eb6834")
        fig.update_layout(barmode="group", height=420, xaxis_title=None, yaxis_title="비율(%)",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ft, use_container_width=True)
    else:
        log_x = st.checkbox("x축 로그스케일 (지출액 등 왜도 큰 변수에 유용)", value=False, key=f"log_{var}")
        group_options = {"(그룹 없음)": None}
        for gcol, gname in [("D_NAT", "국적별"), ("D_MOK", "방한목적별"), ("Q1", "방한목적별"),
                             ("D_GUB", "여행형태별"), ("TYP", "여행형태별"), ("D_AGE", "연령대별")]:
            if gcol in df3.columns and gcol != var:
                group_options[gname] = gcol
        group_choice = st.selectbox("그룹 비교 (박스플롯)", list(group_options.keys()))
        group_col = group_options[group_choice]

        plot_df = df3[[var] + ([group_col] if group_col else [])].dropna(subset=[var]).copy()
        if log_x:
            plot_df = plot_df[plot_df[var] >= 0]
            plot_df["_x"] = np.log1p(plot_df[var])
        else:
            plot_df["_x"] = plot_df[var]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65], vertical_spacing=0.03)

        if group_col:
            gmap = du.get_labels_for(group_col, LABEL_MAP)
            plot_df["_g"] = plot_df[group_col].map(gmap).fillna(plot_df[group_col].astype(str)) if gmap else plot_df[group_col].astype(str)
            top_groups = plot_df["_g"].value_counts().head(10).index.tolist()
            plot_df = plot_df[plot_df["_g"].isin(top_groups)]
            for g in top_groups:
                fig.add_trace(go.Box(x=plot_df.loc[plot_df["_g"] == g, "_x"], name=str(g), orientation="h",
                                      marker_color="#2a78d6", showlegend=False), row=1, col=1)
        else:
            fig.add_trace(go.Box(x=plot_df["_x"], orientation="h", marker_color="#2a78d6", showlegend=False, name=var), row=1, col=1)

        fig.add_trace(go.Histogram(x=plot_df["_x"], nbinsx=50, marker_color="#1baf7a", showlegend=False), row=2, col=1)
        xaxis_title = f"log(1+{var})" if log_x else var
        fig.update_layout(height=520 if group_col else 380, xaxis2_title=xaxis_title, yaxis2_title="빈도")
        st.plotly_chart(fig, use_container_width=True)

        w = df3["weight"]
        st.write(
            f"표본수 N={series.notna().sum():,} · 결측 {series.isna().sum():,}건 · "
            f"비가중평균 {series.mean():.1f} · 가중평균 {du.weighted_mean(series, w):.1f} · "
            f"중앙값 {series.median():.1f} · 최소 {series.min():.1f} · 최대 {series.max():.1f}"
        )

    st.markdown("---")
    st.subheader("2) 연도별 추이 — " + var)
    years_available = sorted(df3["survey_year"].dropna().unique())
    if var == "survey_year":
        st.caption("survey_year 자체는 연도별 추이 대상이 아닙니다. 다른 변수를 선택해주세요.")
    elif len(years_available) < 2:
        st.caption("이 데이터셋은 연도가 1개뿐이라 추이를 볼 수 없습니다.")
    elif categorical:
        tmp = df3[[var, "survey_year", "weight"]].copy()
        tmp["_label"] = tmp[var].map(label_map).fillna(tmp[var].astype(str)) if label_map else tmp[var].astype(str)
        top_cats = tmp.groupby("_label")["weight"].sum().sort_values(ascending=False).head(8).index
        tmp["_label"] = tmp["_label"].where(tmp["_label"].isin(top_cats), "기타")
        pivot = tmp.groupby(["survey_year", "_label"])["weight"].sum().unstack(fill_value=0)
        pivot_pct = (pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
        fig_trend = go.Figure()
        for col in pivot_pct.columns:
            fig_trend.add_trace(go.Scatter(x=pivot_pct.index.astype(str), y=pivot_pct[col],
                                            mode="lines+markers", name=str(col)))
        fig_trend.update_layout(height=380, yaxis_title="가중비율(%)", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("가중치 적용, 비중 상위 8개 범주 외에는 '기타'로 묶었습니다.")
    else:
        tmp = df3[[var, "survey_year", "weight"]].dropna(subset=[var])
        trend = tmp.groupby("survey_year").apply(lambda d: du.weighted_mean(d[var], d["weight"])).round(1)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend.index.astype(str), y=trend.values, mode="lines+markers",
                                        line=dict(color="#2a78d6", width=2), marker=dict(size=8)))
        fig_trend.update_layout(height=340, yaxis_title=f"가중평균 {var}")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("가중평균 기준 연도별 추이입니다.")

    st.markdown("---")
    st.subheader("3) 두 변수 교차표")
    cat_vars_all = [c for c in all_ordered if du.is_categorical(df3[c])]
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        var_a = st.selectbox("변수 A (행)", cat_vars_all, index=0, key="cross_a")
    with cc2:
        remaining_b = [c for c in cat_vars_all if c != var_a]
        default_b = remaining_b.index("D_MOK") if "D_MOK" in remaining_b else (remaining_b.index("Q1") if "Q1" in remaining_b else 0)
        var_b = st.selectbox("변수 B (열)", remaining_b, index=default_b, key="cross_b")
    with cc3:
        weighted_cross = st.checkbox("가중치 적용", value=True, key="cross_w")

    label_a = du.get_labels_for(var_a, LABEL_MAP)
    label_b = du.get_labels_for(var_b, LABEL_MAP)
    tmp_cross = df3[[var_a, var_b, "weight"]].copy()
    tmp_cross["_a"] = tmp_cross[var_a].map(label_a).fillna(tmp_cross[var_a].astype(str)) if label_a else tmp_cross[var_a].astype(str)
    tmp_cross["_b"] = tmp_cross[var_b].map(label_b).fillna(tmp_cross[var_b].astype(str)) if label_b else tmp_cross[var_b].astype(str)

    if weighted_cross:
        cross = tmp_cross.pivot_table(index="_a", columns="_b", values="weight", aggfunc="sum", fill_value=0)
    else:
        cross = pd.crosstab(tmp_cross["_a"], tmp_cross["_b"])
    cross_pct = (cross.div(cross.sum(axis=1), axis=0) * 100).round(1)

    st.markdown(f"**교차표** ({'가중추정치' if weighted_cross else '표본수'})")
    st.dataframe(cross.round(0) if weighted_cross else cross, use_container_width=True)
    st.markdown("**행(변수 A) 기준 비율(%)** — 각 행 내에서 변수 B의 분포")
    st.dataframe(cross_pct, use_container_width=True)

    chart_rows = cross_pct if len(cross_pct) <= 12 else cross_pct.loc[cross.sum(axis=1).sort_values(ascending=False).head(12).index]
    fig_cross = go.Figure()
    for col in chart_rows.columns:
        fig_cross.add_bar(name=str(col), x=chart_rows.index.astype(str), y=chart_rows[col])
    fig_cross.update_layout(barmode="stack", height=420, yaxis_title="비율(%)", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_cross, use_container_width=True)
    if len(cross_pct) > 12:
        st.caption("그래프는 표본수 상위 12개 행만 표시했습니다 (표에는 전체 포함).")

    st.markdown("---")
    st.subheader("4) 지역별 방문 요약 (숙박 기준)")
    region_df = du.region_visit_summary(ds_name3)
    weighted_region = st.checkbox("가중 방문율 기준으로 보기", value=True, key="region_w")
    sort_col = "가중 방문율(%)" if weighted_region else "비가중 방문율(%)"
    region_sorted = region_df.sort_values(sort_col, ascending=False)
    fig_region = go.Figure()
    fig_region.add_bar(x=region_sorted[sort_col], y=region_sorted["지역"], orientation="h", marker_color="#4a3aa7")
    fig_region.update_layout(height=max(320, len(region_sorted) * 26), xaxis_title="방문율(%)",
                              yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_region, use_container_width=True)
    st.dataframe(region_sorted, use_container_width=True)
    st.caption("숙박 기준 방문율입니다 (해당 지역 박TOT > 0인 응답자 비율). 당일 방문(일TOT)은 포함하지 않았습니다.")

    st.markdown("---")
    st.subheader("5) 지출항목 간 상관관계")
    corr_feats = [c for c in du.EXPENDITURE_ITEMS[ds_name3] if c in df3.columns]
    log_corr = st.checkbox("로그(1+x) 변환 후 상관계수 계산", value=True)
    corr_data = df3[corr_feats].fillna(0)
    if log_corr:
        corr_data = np.log1p(corr_data.clip(lower=0))
    corr = corr_data.corr().round(2)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}", textfont={"size": 10}))
    fig_corr.update_layout(height=480)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("6) 결측치 현황")
    miss = df3.isna().sum().to_frame("결측수")
    miss["결측비율(%)"] = (miss["결측수"] / len(df3) * 100).round(2)
    miss = miss.sort_values("결측비율(%)", ascending=False)
    only_missing = st.checkbox("결측 있는 변수만 보기", value=True, key="miss_only")
    show_miss = miss[miss["결측수"] > 0] if only_missing else miss
    st.dataframe(show_miss, use_container_width=True, height=380)
    st.caption(
        f"전체 {df3.shape[1]}개 변수 중 결측이 있는 변수: {(miss['결측수'] > 0).sum()}개. "
        "지역별 방문 변수처럼 '해당 없음'이 결측으로 기록된 구조적 결측이 많습니다 "
        "(그런 변수는 분석 시 0으로 채워야 하는 경우가 많으니 주의하세요)."
    )
