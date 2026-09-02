import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    st.markdown("---")
    st.subheader("연도(2023~2025) 통합 과정에서 제외/변경된 변수")
    st.caption("코드북 워크북(1_codebook_항목정의서_비교.xlsx)의 '데이터별 차이' / '변수명 변경(개요)' 시트 기준")
    diff_df, rename_df, full_compare_df = du.load_diff_sheets()

    excluded = diff_df[diff_df.iloc[:, 7].notna()] if diff_df.shape[1] > 7 else diff_df
    st.markdown("**제외/처리 사유가 명시된 변수**")
    st.dataframe(excluded, use_container_width=True, height=280)

    st.markdown("**연도별 변수명 변경 이력**")
    st.dataframe(rename_df, use_container_width=True, height=280)

    with st.expander("전체 변수 연도별 존재/일치 여부 (전체_비교 시트)"):
        st.dataframe(full_compare_df, use_container_width=True, height=400)

# ============================================================
# TAB 3. 변수 탐색
# ============================================================
with tab3:
    ds_name3 = st.radio("데이터셋 선택", options=["pre", "post"],
                         format_func=lambda x: du.DATASET_LABELS[x], horizontal=True, key="t3_ds")
    df3 = du.load_csv(ds_name3)
    key_numeric = du.KEY_NUMERIC_PRE if ds_name3 == "pre" else du.KEY_NUMERIC_POST
    priority = [c for c in du.KEY_CATEGORICAL + key_numeric if c in df3.columns]
    other_cols = [c for c in df3.columns if c not in priority and c not in ("pnid", "weight", "weight1", "weight2", "weight3", "weight4")]
    all_ordered = priority + sorted(other_cols)

    st.subheader("1) 변수별 분포 탐색")
    var = st.selectbox("변수 선택", all_ordered, index=0)

    desc, dtype = du.get_var_desc(var, DESC_MAP, DTYPE_MAP)
    label_map = du.get_labels_for(var, LABEL_MAP)
    st.info(f"**{var}** · 데이터형태: {dtype or '미상'}\n\n{desc or '설명 정보 없음'}")

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
    st.subheader("2) 지출항목 간 상관관계")
    corr_feats = [c for c in du.CLUSTER_FEATURES[ds_name3] if c != "M일HAP" and c in df3.columns]
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
    st.subheader("3) 군집분석 — 체류일수 × 지출항목 (로그변환, K=4)")
    st.caption("입력변수: " + ", ".join(du.CLUSTER_FEATURES[ds_name3]) + " · log(1+x) 변환 후 표준화 → K-means(K=4, 가중치는 알고리즘에 미반영, 해석 시에만 가중비율 병기)")

    @st.cache_data(show_spinner="군집분석(K=4) 계산 중...")
    def compute_clusters(name: str):
        d = du.load_csv(name)
        feats = du.CLUSTER_FEATURES[name]
        spend_feats = [f for f in feats if f != "M일HAP"]
        sub = d[feats + ["weight", "D_NAT"]].copy()
        mok_col = "D_MOK" if "D_MOK" in d.columns else "Q1"
        gub_col = "D_GUB" if "D_GUB" in d.columns else "TYP"
        sub[mok_col] = d[mok_col]
        sub[gub_col] = d[gub_col]
        sub["총액1인TOT2"] = d["총액1인TOT2"]
        sub[spend_feats] = sub[spend_feats].fillna(0)
        sub = sub.dropna(subset=["M일HAP", "weight"])

        X = np.log1p(sub[feats].clip(lower=0))
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        sub["cluster"] = km.fit_predict(Xs)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
        sub["pc1"], sub["pc2"] = coords[:, 0], coords[:, 1]
        return sub, feats, mok_col, gub_col

    sub, feats, mok_col, gub_col = compute_clusters(ds_name3)

    size_tbl = sub.groupby("cluster").agg(표본수=("weight", "size"), 가중치합=("weight", "sum"),
                                           평균체류일수=("M일HAP", "mean"), 평균총지출=("총액1인TOT2", "mean"))
    size_tbl["가중비율(%)"] = (size_tbl["가중치합"] / size_tbl["가중치합"].sum() * 100).round(1)
    size_tbl["평균체류일수"] = size_tbl["평균체류일수"].round(1)
    size_tbl["평균총지출"] = size_tbl["평균총지출"].round(0)

    nat_labels_map = du.get_labels_for("D_NAT", LABEL_MAP)
    sub["_nat_label"] = sub["D_NAT"].map(nat_labels_map).fillna(sub["D_NAT"].astype(str))
    top_nat = sub.groupby("cluster")["_nat_label"].agg(lambda s: s.value_counts().idxmax())
    mok_labels_map = du.get_labels_for(mok_col, LABEL_MAP)
    sub["_mok_label"] = sub[mok_col].map(mok_labels_map).fillna(sub[mok_col].astype(str))
    top_mok = sub.groupby("cluster")["_mok_label"].agg(lambda s: s.value_counts().idxmax())

    size_tbl["주요 국적"] = top_nat
    size_tbl["주요 방한목적"] = top_mok
    size_tbl = size_tbl.drop(columns=["가중치합"])
    st.dataframe(size_tbl, use_container_width=True)

    profile = sub.groupby("cluster")[feats].mean().round(1)
    st.markdown("**군집별 평균 프로파일 (원 단위, 로그변환 이전 값)**")
    st.dataframe(profile, use_container_width=True)

    sample = sub.sample(min(5000, len(sub)), random_state=42)
    fig_pca = go.Figure()
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for c in sorted(sample["cluster"].unique()):
        s = sample[sample["cluster"] == c]
        fig_pca.add_trace(go.Scatter(x=s["pc1"], y=s["pc2"], mode="markers", name=f"군집 {c}",
                                      marker=dict(size=5, color=colors[c % 4], opacity=0.5)))
    fig_pca.update_layout(height=460, xaxis_title="PC1", yaxis_title="PC2",
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("PCA 2차원 투영 시각화 (5,000건 샘플)")

    gub_labels_map = du.get_labels_for(gub_col, LABEL_MAP)
    sub["_gub_label"] = sub[gub_col].map(gub_labels_map).fillna(sub[gub_col].astype(str))
    cross = pd.crosstab(sub["cluster"], sub["_gub_label"], normalize="index") * 100
    fig_cross = go.Figure()
    for col in cross.columns:
        fig_cross.add_bar(name=str(col), x=[f"군집 {i}" for i in cross.index], y=cross[col].round(1))
    fig_cross.update_layout(barmode="stack", height=360, yaxis_title="비율(%)",
                             legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_cross, use_container_width=True)
    st.caption("군집별 여행형태 구성비")
