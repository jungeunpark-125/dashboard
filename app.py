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

    with st.expander("(참고) 2023~2025 통합 시 KTO 코드북 처리 메모 — 잘 안 쓰지만 남겨둠"):
        diff_df, rename_df, full_compare_df = du.load_diff_sheets()
        st.markdown("**제외/처리 사유가 명시된 변수** (2023~2025 내부 통합 과정, pre/post 비교와는 무관)")
        st.dataframe(diff_df, use_container_width=True, height=240)
        st.markdown("**연도별 변수명 변경 이력**")
        st.dataframe(rename_df, use_container_width=True, height=240)

# ============================================================
# TAB 3. 변수 탐색
# ============================================================
with tab3:
    ds_name3 = st.radio("데이터셋 선택", options=["pre", "post"],
                         format_func=lambda x: du.DATASET_LABELS[x], horizontal=True, key="t3_ds")
    df3 = du.load_csv(ds_name3)

    with st.expander("가중치(weight) 는 이렇게 처리했습니다 — 펼쳐서 보기"):
        st.markdown(
            "- `weight1`~`weight4` = **분기별 가중치**이지만 **모든 연도에 있는 게 아닙니다.** 실제로 값이 존재하는 건 "
            "**2024·2025년뿐**이고, 그 안에서도 응답자 1명당 자신이 응답한 분기 컬럼에만 값이 있고 나머지 3개는 결측입니다. "
            "**2019·2023년은 weight1~4 컬럼 자체가 전부 결측**입니다 (그 해에는 분기별 가중치를 별도 산출하지 않은 것으로 보입니다).\n"
            "- `weight` = **연간 통합 가중치**로, **2019~2025 전 연도에 걸쳐 결측이 0건**입니다. "
            "이 대시보드의 모든 '가중' 통계는 연도와 무관하게 **이 `weight` 컬럼만 사용**하기 때문에, 위의 연도별 weight1~4 결측 여부와 무관하게 "
            "모든 연도의 가중 계산이 정상적으로 이뤄집니다 (weight1~4는 분기 단위 세부분석이 필요할 때만 쓰는 보조 변수이며, 애초에 전 연도에 존재하지 않아 "
            "이 대시보드에서는 사용하지 않습니다).\n"
            "- **계산식**: 가중평균 = Σ(x·weight) / Σweight, 가중비율(%) = (카테고리별 weight 합) / (전체 weight 합) × 100.\n"
            "- **비가중(표본) vs 가중(모집단추정) 이 왜 다른가**: 조사 표본설계상 국적별로 표본을 고르게 뽑지 않고(소규모 국가를 상대적으로 더 뽑는 식) "
            "실제 방한객 규모에 맞춰 사후 보정한 게 `weight`입니다. 그래서 중국·일본처럼 실제 비중이 큰 국적은 표본비율보다 가중비율이 훨씬 높게 나옵니다 "
            "— 즉 **모집단(실제 방한 외국인 전체) 추정치로 해석하려면 항상 가중값을 봐야 합니다.**\n"
            "- 참고로 `weight`의 **단위(스케일)도 연도별로 다릅니다** — 2019년은 평균 1 안팎(비율형 보정계수)인 반면 2023~2025년은 평균 수백~천 단위(응답자 1명이 대표하는 인원수 규모)입니다. "
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

    with st.expander("군집분석은 어떻게, 왜 이렇게 했나요? — 펼쳐서 보기"):
        st.markdown(
            "**입력변수**: 체류일수(M일HAP) + 주요 지출항목(위 '입력변수' 참고). 지출 세부 20여 개 항목 중 "
            "0값이 압도적으로 많은 소액/희소 항목은 빼고, 지출 구조를 대표하는 주요 항목만 선택했습니다.\n\n"
            "**왜 로그변환을 했나**: 지출액은 오른쪽으로 크게 치우친 분포입니다(소수의 초고액 지출자가 존재). "
            "원 단위로 그대로 군집분석에 넣으면 이 극단값들이 '거리' 계산을 지배해서 대부분의 평범한 응답자가 "
            "제대로 구분되지 않습니다. `log(1+x)`로 압축한 뒤 평균 0·분산 1로 표준화해야 모든 변수가 "
            "비슷한 스케일로 군집 형성에 기여합니다.\n\n"
            "**K=4는 왜**: 사용자가 지정한 고정값입니다(요청 시 4개로 진행). 최적 K를 찾는 통계적 절차(엘보우/실루엣)를 "
            "거친 값이 아니라는 점은 참고해주세요.\n\n"
            "**가중치는 어떻게 반영했나**: K-means 알고리즘 자체는 표본 간 기하학적 거리로 군집을 나누는 기법이라 "
            "설문 가중치를 직접 반영하는 표준적인 방법이 없습니다. 그래서 **군집을 나누는 계산에는 가중치를 쓰지 않고**, "
            "그 대신 각 군집이 실제 모집단에서 차지하는 비중을 알 수 있도록 **결과 표에 가중비율(%)을 별도로 병기**했습니다.\n\n"
            "**각 군집이 '무엇'인지는 미리 정해진 게 아닙니다** — K-means는 그냥 비슷한 응답자끼리 4개 그룹으로 묶을 뿐, "
            "'단기형', '장기 고지출형' 같은 이름은 없습니다. 아래 표의 평균 체류일수·평균 총지출·주요 국적·주요 방한목적을 "
            "보고 **사람이 사후적으로 해석**하는 것이며, `특징` 컬럼은 그 해석을 돕기 위해 전체 평균 대비 체류일수·지출 수준을 "
            "자동으로 요약한 것입니다."
        )

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
        loadings = pd.DataFrame(pca.components_.T, index=feats, columns=["PC1", "PC2"]).round(2)
        return sub, feats, mok_col, gub_col, pca.explained_variance_ratio_, loadings

    sub, feats, mok_col, gub_col, explained_var, loadings = compute_clusters(ds_name3)

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

    day_med, spend_med = size_tbl["평균체류일수"].median(), size_tbl["평균총지출"].median()
    def _label(row):
        day_tag = "장기" if row["평균체류일수"] >= day_med else "단기"
        spend_tag = "고지출" if row["평균총지출"] >= spend_med else "저지출"
        return f"{day_tag}·{spend_tag}"
    size_tbl["특징(자동요약)"] = size_tbl.apply(_label, axis=1)
    st.dataframe(size_tbl, use_container_width=True)

    profile = sub.groupby("cluster")[feats].mean().round(1)
    st.markdown("**군집별 평균 프로파일 (원 단위, 로그변환 이전 값)**")
    st.dataframe(profile, use_container_width=True)

    with st.expander("PCA 산점도는 무엇을 보여주나요? PC1·PC2가 뭔가요? — 펼쳐서 보기"):
        st.markdown(
            f"군집분석은 실제로는 {len(feats)}개 변수(고차원 공간)에서 이루어지지만, 사람이 눈으로 확인할 수 있도록 "
            "**PCA(주성분분석)로 정보 손실을 최소화하면서 2개의 축(PC1, PC2)에 압축 투영**한 것이 아래 산점도입니다.\n\n"
            "- **PC1, PC2는 원 변수 하나하나가 아니라, 여러 변수를 섞은 합성축(가중합)**입니다. 예를 들어 PC1이 "
            "'전체적인 지출·체류 규모'와, PC2가 '지출 항목의 구성(예: 쇼핑 중심 vs 숙박 중심)'과 관련될 수 있는데, "
            "정확히 무엇과 관련되는지는 아래 '변수별 기여도(loading)' 표의 절댓값이 큰 변수로 판단합니다.\n"
            f"- 이 두 축이 원래 {len(feats)}개 변수가 가진 정보(분산)의 **{explained_var.sum()*100:.1f}%**를 담고 있습니다"
            f" (PC1 {explained_var[0]*100:.1f}%, PC2 {explained_var[1]*100:.1f}%). 100%가 아니므로 2D 그림은 "
            "'근사적인 요약'이지 원 데이터 그대로는 아닙니다.\n\n"
            "**이 그림으로 알 수 있는 것**: ① 군집(색)들이 평면에서 잘 갈라져 보이면 군집 구분이 뚜렷하다는 뜻이고, "
            "서로 겹쳐 보이면 그 두 군집은 실제로도 특성이 비슷하다는 뜻입니다. ② 점들이 서로 가까울수록 "
            "체류·지출 패턴이 비슷한 응답자라는 뜻입니다."
        )
        st.markdown("**변수별 기여도(loading)** — 절댓값이 클수록 해당 축(PC1/PC2)에 그 변수가 많이 반영됨")
        st.dataframe(loadings, use_container_width=True)

    sample = sub.sample(min(5000, len(sub)), random_state=42)
    fig_pca = go.Figure()
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for c in sorted(sample["cluster"].unique()):
        s = sample[sample["cluster"] == c]
        fig_pca.add_trace(go.Scatter(x=s["pc1"], y=s["pc2"], mode="markers", name=f"군집 {c}",
                                      marker=dict(size=5, color=colors[c % 4], opacity=0.5)))
    fig_pca.update_layout(height=460,
                           xaxis_title=f"PC1 (분산의 {explained_var[0]*100:.1f}% 설명)",
                           yaxis_title=f"PC2 (분산의 {explained_var[1]*100:.1f}% 설명)",
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
