"""Credit Portfolio Segmentation & Risk Pattern Analysis — Streamlit dashboard.

Place this file in Credit_Portfolio_Segmentation/app.py alongside project_outputs/.
Required: project_outputs/credit_segment_scorecard.csv
Run: streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Portfolio Intelligence", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

YELLOW = "#F4B400"
DARK_YELLOW = "#D18C00"
INK = "#202735"
MUTED = "#687385"
BG = "#FFFDF5"
COLORS = ["#F4B400", "#F8CD54", "#EAA11C", "#D58B10", "#FFE29A"]

st.markdown("""
<style>
:root { --accent:#F4B400; }
.stApp {background: #FFFDF5; color:#202735;}
[data-testid="stSidebar"] {display:none;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {color:#202735;}
.block-container {padding-top:1rem; padding-bottom:3rem; max-width:1650px;}
h1,h2,h3 {color:#202735 !important; letter-spacing:-.025em;}
h1 {font-size:2.05rem !important;}
[data-testid="stRadio"] [role="radiogroup"] {background:#FFF4D1;padding:8px;border-radius:12px;}
[data-testid="stRadio"] label {font-weight:600;}
.section-title {font-size:1.05rem;font-weight:800;color:#202735;margin-bottom:.5rem;}
.right-panel {background:#FFFFFF;border:1px solid #F1DCA2;border-radius:12px;padding:18px 20px;margin-bottom:14px;line-height:1.55;}
.right-panel h4 {margin:0 0 8px;color:#202735;}
.right-panel p {margin:0 0 10px;}
.right-panel ul {padding-left:20px;margin:0;}
h2 {font-size:1.45rem !important;}
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #F2DEAC; border-top:4px solid #F4B400; border-radius:13px; padding:16px 17px; box-shadow:0 3px 12px rgba(103,75,8,.045);}
[data-testid="stMetricLabel"] {color:#687385;}
[data-testid="stMetricValue"] {color:#202735; font-weight:750;}
.insight {background:#FFF8E2; border-left:4px solid #F4B400; border-radius:8px; padding:14px 18px; margin:12px 0 22px; color:#394150;}
.section-caption {color:#687385; font-size:.94rem; margin-bottom:1rem;}
.small-note {font-size:.83rem; color:#687385;}
[data-testid="stDataFrame"] {border:1px solid #F2DEAC; border-radius:9px; overflow:hidden;}
hr {border-color:#F2DEAC !important;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
SCORECARD_PATH = ROOT / "project_outputs" / "credit_segment_scorecard.csv"
REQUIRED = ["Cluster", "Segment_Name", "Borrowers", "Borrower_Share_%", "Defaults", "Default_Rate_%", "Relative_Default_Index", "Default_Share_%", "Avg_Original_Loan", "Total_Original_Principal", "Principal_Share_%"]

@st.cache_data(show_spinner=False)
def load_scorecard(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)

if not SCORECARD_PATH.is_file():
    st.error(f"Required scorecard not found: {SCORECARD_PATH}")
    st.info("Keep app.py in your Credit_Portfolio_Segmentation folder and the CSV inside project_outputs.")
    st.stop()

try:
    df = load_scorecard(str(SCORECARD_PATH), SCORECARD_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read scorecard: {exc}")
    st.stop()
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing required scorecard columns: {missing}")
    st.stop()
if df.empty or df["Cluster"].duplicated().any():
    st.error("Scorecard must have one unique row per segment.")
    st.stop()
for col in ["Borrowers", "Defaults", "Borrower_Share_%", "Default_Rate_%", "Relative_Default_Index", "Default_Share_%", "Avg_Original_Loan", "Total_Original_Principal", "Principal_Share_%"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[REQUIRED].isna().any().any() or (df["Borrowers"] <= 0).any():
    st.error("The scorecard has invalid or missing required values.")
    st.stop()
df = df.sort_values("Cluster").reset_index(drop=True)

TOTAL = int(df["Borrowers"].sum())
DEFAULTS = int(df["Defaults"].sum())
RATE = DEFAULTS / TOTAL * 100
PRINCIPAL = float(df["Total_Original_Principal"].sum())

# Benchmark results come from the 5,000-row training comparison experiment.
BENCHMARK = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical", "DBSCAN"],
    "Clusters": [5, 5, 9],
    "Silhouette": [0.1418, 0.0839, 0.0500],
    "Fit_seconds": [0.0778, 0.2148, 0.0332],
    "Noise_%": [0.0, 0.0, 16.18],
    "Evaluation_population": ["Sampled from all 5,000", "Sampled from all 5,000", "Sampled from 4,191 non-noise"]
})
ANOMALY = pd.DataFrame({
    "Group": ["Clustered", "Noise"],
    "Borrowers": [4191, 809],
    "Historical defaults": [799, 138],
    "Historical default rate (%)": [19.06, 17.06],
    "Average income": [71514.56, 93058.75],
    "Average DTI": [18.34, 16.10],
    "Average original loan": [14463.99, 13761.16],
    "Average FICO": [692.34, 718.29]
})

def plot_style(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color=INK, family="Arial"),
        margin=dict(l=15, r=20, t=48, b=35),
        title_font=dict(size=16, color=INK),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E8E3D9")
    fig.update_yaxes(gridcolor="#F0EDE5", zeroline=False)
    return fig

def insight(text):
    st.markdown(f'<div class="insight"><b>Business insight</b><div style="margin-top:5px">{text}</div></div>', unsafe_allow_html=True)

def page_title(title, subtitle):
    st.title(title)
    st.markdown(f'<div class="section-caption">{subtitle}</div>', unsafe_allow_html=True)


# Optional borrower-level data, required for real Year / Purpose / Status filters.
# The five-row scorecard has no year, purpose or borrower-level outcome columns.
DETAIL_PATH = ROOT / "project_outputs" / "credit_borrower_dashboard.csv"
@st.cache_data(show_spinner=False)
def load_details(path, modified):
    return pd.read_csv(path, low_memory=False)

has_details = DETAIL_PATH.is_file()
detail = None
if has_details:
    try:
        detail = load_details(str(DETAIL_PATH), DETAIL_PATH.stat().st_mtime)
        if not {"Cluster", "Default", "loan_amnt"}.issubset(detail.columns):
            st.warning("Borrower-level file needs Cluster, Default and loan_amnt columns. Using the verified scorecard.")
            detail = None
        else:
            detail["Cluster"] = pd.to_numeric(detail["Cluster"], errors="coerce")
            detail["Default"] = pd.to_numeric(detail["Default"], errors="coerce")
            detail["loan_amnt"] = pd.to_numeric(detail["loan_amnt"], errors="coerce")
            detail = detail.dropna(subset=["Cluster", "Default", "loan_amnt"])
            detail = detail[detail["Cluster"].isin(df["Cluster"])]
            detail = detail[detail["Default"].isin([0,1])]
            if detail.empty:
                st.warning("Borrower-level file contains no valid rows. Using the scorecard.")
                detail = None
    except Exception as exc:
        st.warning(f"Could not read optional borrower-level file: {exc}")
        detail = None

st.markdown("# 🟡 Credit Portfolio Intelligence")
st.caption("Credit Portfolio Segmentation & Risk Pattern Analysis  |  Executive decision-support dashboard")
page = st.radio("Dashboard pages", ["Executive Overview", "Segment Intelligence", "Model Evaluation", "Anomaly Analysis", "Borrower Segmentation"], horizontal=True, label_visibility="collapsed")
st.divider()

# Scorecard is the verified source for the complete historical test portfolio.
# Detailed filters are applied only when an aligned borrower-level export exists.
view = df.copy()
filter_active = False
if page in ("Executive Overview", "Segment Intelligence"):
    st.markdown('<div class="section-title">Portfolio filters</div>', unsafe_allow_html=True)
    if detail is not None:
        f1, f2, f3 = st.columns([1, 1.5, 1])
        with f1:
            if "issue_year" in detail.columns:
                years = sorted(detail["issue_year"].dropna().astype(str).unique().tolist())
                selected_years = st.multiselect("Loan issue year", years, default=years)
            else:
                selected_years = None
                st.caption("Year filter requires issue_year in the detailed CSV")
        with f2:
            if "purpose" in detail.columns:
                purposes = sorted(detail["purpose"].dropna().astype(str).unique().tolist())
                selected_purposes = st.multiselect("Loan purpose", purposes, default=purposes)
            else:
                selected_purposes = None
                st.caption("Purpose filter requires purpose in the detailed CSV")
        with f3:
            status = st.selectbox("Historical outcome", ["All borrowers", "Default", "Non-default"])
        filtered = detail.copy()
        if selected_years is not None:
            filtered = filtered[filtered["issue_year"].astype(str).isin(selected_years)]
        if selected_purposes is not None:
            filtered = filtered[filtered["purpose"].astype(str).isin(selected_purposes)]
        if status == "Default":
            filtered = filtered[filtered["Default"] == 1]
        elif status == "Non-default":
            filtered = filtered[filtered["Default"] == 0]
        filter_active = True
        if filtered.empty:
            st.warning("No borrowers match these filters. Change the selection.")
            st.stop()
        view = filtered.groupby("Cluster", as_index=False).agg(
            Borrowers=("Default", "size"), Defaults=("Default", "sum"),
            Total_Original_Principal=("loan_amnt", "sum"), Avg_Original_Loan=("loan_amnt", "mean")
        ).merge(df[["Cluster", "Segment_Name"]], on="Cluster", how="left")
        total_n = view["Borrowers"].sum()
        total_d = view["Defaults"].sum()
        total_p = view["Total_Original_Principal"].sum()
        view["Borrower_Share_%"] = 100 * view["Borrowers"] / total_n
        view["Default_Rate_%"] = 100 * view["Defaults"] / view["Borrowers"]
        view["Default_Share_%"] = 100 * view["Defaults"] / total_d if total_d else 0
        view["Principal_Share_%"] = 100 * view["Total_Original_Principal"] / total_p if total_p else 0
        view["Relative_Default_Index"] = view["Default_Rate_%"] / (100*total_d/total_n) if total_d else 0
        st.caption(f"Filtered historical cohort: {len(filtered):,} borrowers. Outcome filters change the denominator; do not interpret the resulting rate as the full-portfolio default rate.")
    else:
        st.info("**Year, loan-purpose and default-status filters require borrower-level data.** The current saved scorecard contains only five aggregated segments, so these filters cannot be calculated accurately yet. Export `credit_borrower_dashboard.csv` with Cluster, Default, loan_amnt, issue_year and purpose to activate them. The segment selector below is available now.")

# Aggregate segment-rate filter is available with or without borrower-level data.
if page in ("Executive Overview", "Segment Intelligence"):
    low_rate, high_rate = st.slider(
        "Segment historical default-rate range (%)",
        min_value=0.0, max_value=100.0, value=(0.0, 100.0), step=0.5,
        help="Select segments by their observed historical default rate. This is not an individual borrower default-probability filter."
    )
    view = view[view["Default_Rate_%"].between(low_rate, high_rate)].copy()
    if view.empty:
        st.warning("No segments match the selected default-rate range.")
        st.stop()
    if low_rate > 0 or high_rate < 100:
        st.caption("Segment-rate filter active: KPI totals now cover only the selected segments.")

n = int(view["Borrowers"].sum())
d = int(view["Defaults"].sum())
r = 100*d/n if n else 0
principal = float(view["Total_Original_Principal"].sum())

if page == "Executive Overview":
    page_title("Executive Overview", "Historical test portfolio • business performance, concentrations and borrower segmentation")
    a,b,c,e = st.columns(4)
    a.metric("Borrowers in view", f"{n:,}")
    b.metric("Historical defaults", f"{d:,}")
    c.metric("Historical default rate", f"{r:.2f}%")
    e.metric("Original loan principal", f"{principal/1e9:.2f}B")
    st.caption("Original principal uses unverified source currency units and is not outstanding exposure or realized credit loss.")
    main, side = st.columns([3.2, 1.15], gap="large")
    with main:
        a,b = st.columns(2, gap="medium")
        with a:
            fig = px.bar(view.sort_values("Borrowers"), x="Borrowers", y="Segment_Name", orientation="h", text="Borrowers", title="Borrower distribution", color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            st.plotly_chart(plot_style(fig, 390), use_container_width=True)
        with b:
            fig = px.bar(view.sort_values("Default_Rate_%"), x="Default_Rate_%", y="Segment_Name", orientation="h", text="Default_Rate_%", title="Historical default rate by segment", color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
            fig.add_vline(x=r, line_dash="dash", line_color=DARK_YELLOW)
            st.plotly_chart(plot_style(fig, 390), use_container_width=True)
        a,b = st.columns(2, gap="medium")
        with a:
            fig = px.pie(view, names="Segment_Name", values="Total_Original_Principal", hole=.60, title="Original principal concentration", color_discrete_sequence=COLORS)
            fig.update_traces(textinfo="percent")
            st.plotly_chart(plot_style(fig, 375), use_container_width=True)
        with b:
            fig = px.bar(view.sort_values("Default_Share_%", ascending=False), x="Segment_Name", y="Default_Share_%", text="Default_Share_%", title="Share of historical defaults", color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(plot_style(fig, 375), use_container_width=True)
    with side:
        st.markdown('''<div class="right-panel"><h4>Project objective</h4><p>Group borrowers by financial and demographic characteristics, compare historical default patterns, identify concentrations of original loan principal, and examine unusual profiles.</p><p><b>Methods:</b> K-Means, Hierarchical, DBSCAN, PCA, t-SNE and UMAP.</p></div>''', unsafe_allow_html=True)
        if not view.empty:
            high = view.loc[view["Default_Rate_%"].idxmax()]
            concentrated = view.loc[view["Principal_Share_%"].idxmax()]
            biggest = view.loc[view["Borrowers"].idxmax()]
            st.markdown(f'''<div class="right-panel"><h4>Key takeaways</h4><ul><li><b>Highest observed default rate:</b> {high['Segment_Name']} ({high['Default_Rate_%']:.2f}%).</li><li><b>Largest principal concentration:</b> {concentrated['Segment_Name']} ({concentrated['Principal_Share_%']:.2f}%).</li><li><b>Largest borrower group:</b> {biggest['Segment_Name']} ({int(biggest['Borrowers']):,}).</li><li><b>Portfolio rate:</b> {r:.2f}% for the selected cohort.</li></ul></div>''', unsafe_allow_html=True)
        st.markdown('''<div class="right-panel"><h4>Executive interpretation</h4><p>Segment default rates and original-principal shares measure different dimensions of the portfolio. Higher original principal does not establish higher realized loss.</p><p><b>Important:</b> Different loan vintages have unequal time to develop a default. Results are descriptive, not predictions or credit approval rules.</p></div>''', unsafe_allow_html=True)
    with st.expander("Detailed segment scorecard"):
        st.dataframe(view.round(3), use_container_width=True, hide_index=True)

elif page == "Segment Intelligence":
    page_title("Segment Intelligence", "Compare borrower segments and investigate historical portfolio characteristics")
    choice = st.selectbox("Select borrower segment", view["Segment_Name"].tolist())
    seg = view.loc[view["Segment_Name"] == choice].iloc[0]
    a,b,c,e = st.columns(4)
    a.metric("Borrowers", f"{int(seg['Borrowers']):,}")
    b.metric("Borrower share", f"{seg['Borrower_Share_%']:.2f}%")
    c.metric("Historical default rate", f"{seg['Default_Rate_%']:.2f}%")
    e.metric("Principal share", f"{seg['Principal_Share_%']:.2f}%")
    main,side = st.columns([3.2,1.15],gap="large")
    with main:
        a,b=st.columns(2)
        with a:
            fig=go.Figure(go.Bar(x=["Selected segment","Portfolio"],y=[seg["Default_Rate_%"],r],marker_color=[YELLOW,"#AAB2BE"],text=[f"{seg['Default_Rate_%']:.2f}%",f"{r:.2f}%"],textposition="outside"))
            fig.update_layout(title="Historical default rate vs portfolio",yaxis_title="Default rate (%)")
            st.plotly_chart(plot_style(fig),use_container_width=True)
        with b:
            fig=go.Figure(go.Bar(x=["Borrower share","Principal share","Default share"],y=[seg["Borrower_Share_%"],seg["Principal_Share_%"],seg["Default_Share_%"]],marker_color=COLORS[:3],text=[f"{seg[k]:.1f}%" for k in ["Borrower_Share_%","Principal_Share_%","Default_Share_%"]],textposition="outside"))
            fig.update_layout(title="Selected segment contribution",yaxis_title="Share (%)")
            st.plotly_chart(plot_style(fig),use_container_width=True)
        st.subheader("Segment comparison")
        st.dataframe(view[["Segment_Name","Borrowers","Borrower_Share_%","Default_Rate_%","Relative_Default_Index","Principal_Share_%"]].round(2),hide_index=True,use_container_width=True)
    with side:
        st.markdown(f'''<div class="right-panel"><h4>Selected segment</h4><p><b>{choice}</b></p><p>Relative historical default index: <b>{seg['Relative_Default_Index']:.2f}</b> (1.00 = current cohort average).</p></div>''',unsafe_allow_html=True)
        st.markdown('''<div class="right-panel"><h4>How to interpret</h4><p>Segment names summarize group characteristics, not eligibility decisions. Differences in historical default rates may reflect borrower composition and loan observation periods.</p></div>''',unsafe_allow_html=True)

elif page == "Model Evaluation":
    page_title("Model Evaluation", "Comparison on 5,000 training borrowers using 24 preprocessed features")
    a,b,c=st.columns(3)
    a.metric("Primary model","K-Means")
    b.metric("Silhouette","0.1418")
    c.metric("Segments","5")
    main,side=st.columns([3.2,1.15],gap="large")
    with main:
        a,b=st.columns(2)
        with a:
            fig=px.bar(BENCHMARK,x="Algorithm",y="Silhouette",text="Silhouette",title="Silhouette comparison",color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
            st.plotly_chart(plot_style(fig),use_container_width=True)
        with b:
            fig=px.bar(BENCHMARK,x="Algorithm",y="Fit_seconds",text="Fit_seconds",title="Measured fit time",color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
            st.plotly_chart(plot_style(fig),use_container_width=True)
        st.dataframe(BENCHMARK,hide_index=True,use_container_width=True)
        st.subheader("Dimensionality reduction")
        a,b,c=st.columns(3)
        a.metric("PCA • 7 PCs","93.52% variance")
        b.metric("PCA • 2 PCs","45.52% variance")
        c.metric("t-SNE KL divergence","1.5981")
        st.caption("PCA + K-Means silhouette on 7 PCs: 0.1543, measured in a different feature space. UMAP and t-SNE were used for visualization, not prediction.")
        st.info("PCA, t-SNE and UMAP scatter plots need saved point-level embeddings. They are not included in the five-row scorecard.")
    with side:
        st.markdown('''<div class="right-panel"><h4>Model selection</h4><p>K-Means achieved the highest observed silhouette among the compared algorithms and provided five interpretable business segments.</p></div>''',unsafe_allow_html=True)
        st.markdown('''<div class="right-panel"><h4>Key limitations</h4><ul><li>Low silhouette scores indicate weak separation.</li><li>DBSCAN silhouette excludes noise, so it is not strictly comparable.</li><li>Single-run fitting times are not production benchmarks.</li><li>2D visualization does not validate model accuracy.</li></ul></div>''',unsafe_allow_html=True)

elif page == "Borrower Segmentation":
    page_title(
        "Borrower Segmentation",
        "Assign a borrower to one of five K-Means segments using the saved model"
    )
    st.info(
        "This tool assigns a descriptive borrower segment. It does not estimate an "
        "individual probability of default or recommend credit approval."
    )

    import joblib

    model_path = ROOT / "models" / "final_kmeans_model.joblib"
    prep_path = ROOT / "models" / "final_preprocessing.joblib"
    module_path = ROOT / "borrower_preprocessing.py"

    missing_files = [
        str(p.relative_to(ROOT))
        for p in (model_path, prep_path, module_path)
        if not p.is_file()
    ]
    if missing_files:
        st.error("Required deployment files are missing: " + ", ".join(missing_files))
        st.stop()

    @st.cache_resource(show_spinner=False)
    def load_segmentation_model(path, modified):
        return joblib.load(path)

    try:
        from borrower_preprocessing import preprocess_borrowers
        bundle = load_segmentation_model(
            str(model_path), model_path.stat().st_mtime
        )
        fitted_model = bundle["model"]
        segment_names = bundle["segment_mapping"]
    except Exception as exc:
        st.error(f"Unable to load the segmentation pipeline: {exc}")
        st.stop()

    left, right = st.columns([2.3, 1], gap="large")
    with left:
        st.subheader("Borrower details")
        with st.form("borrower_segmentation_form"):
            c1, c2 = st.columns(2)
            with c1:
                income = st.number_input(
                    "Annual income (dataset units)", min_value=1.0,
                    value=75000.0, step=1000.0
                )
                loan_amount = st.number_input(
                    "Original loan amount (dataset units)", min_value=1.0,
                    value=15000.0, step=500.0
                )
                employment = st.selectbox(
                    "Employment length",
                    ["< 1 year"] + [f"{i} year" if i == 1 else f"{i} years"
                                    for i in range(1, 10)] + ["10+ years", "NI"],
                    index=5
                )
                ownership = st.selectbox(
                    "Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"]
                )
            with c2:
                fico = st.number_input(
                    "FICO score", min_value=300, max_value=850, value=710
                )
                dti = st.number_input(
                    "Debt-to-income ratio (DTI)", min_value=0.0,
                    value=18.0, step=0.5,
                    help="Use the same DTI units as the training dataset."
                )
                purpose = st.selectbox(
                    "Loan purpose",
                    ["car", "credit_card", "debt_consolidation", "educational",
                     "home_improvement", "house", "major_purchase", "medical",
                     "moving", "other", "renewable_energy", "small_business",
                     "vacation", "wedding"],
                    index=2
                )
            submitted = st.form_submit_button(
                "Assign borrower segment", type="primary", use_container_width=True
            )

        if submitted:
            raw = pd.DataFrame([{
                "revenue": income,
                "dti_n": dti,
                "loan_amnt": loan_amount,
                "fico_n": fico,
                "emp_length": employment,
                "home_ownership_n": ownership,
                "purpose": purpose
            }])
            try:
                features = preprocess_borrowers(raw)
                if features.columns.tolist() != bundle["feature_columns"]:
                    raise ValueError("Preprocessed feature order differs from saved model.")
                cluster = int(fitted_model.predict(features)[0])
                segment = segment_names[cluster]
                st.session_state["borrower_segment_result"] = (cluster, segment)
            except Exception as exc:
                st.session_state.pop("borrower_segment_result", None)
                st.error(f"Could not assign segment: {exc}")

        result = st.session_state.get("borrower_segment_result")
        if result is not None:
            cluster, segment = result
            st.success(f"Assigned segment: Cluster {cluster} — {segment}")
            match = df.loc[df["Cluster"].astype(int) == cluster]
            if not match.empty:
                historical = match.iloc[0]
                a, b, c = st.columns(3)
                a.metric("Historical segment default rate",
                         f"{historical['Default_Rate_%']:.2f}%")
                b.metric("Historical test borrowers",
                         f"{int(historical['Borrowers']):,}")
                c.metric("Historical portfolio average", f"{RATE:.2f}%")
                st.caption(
                    "Historical segment statistics are from the 2017–2018 test "
                    "portfolio, not a prediction of this borrower's outcome."
                )

    with right:
        st.markdown(
            '<div class="right-panel"><h4>How it works</h4>'
            '<p>Raw borrower details are transformed using the saved income '
            'logarithm, DTI cap, employment mapping, numerical scaler and '
            'categorical encoder. The fitted K-Means model assigns a cluster.</p>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="right-panel"><h4>Interpretation</h4>'
            '<p>The segment name summarizes a historical group. Its observed '
            'default rate is descriptive and must not be treated as an '
            'individual credit-risk probability.</p>'
            '</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="right-panel"><h4>Model limitations</h4>'
            '<p>Cluster separation is relatively weak. Training-data '
            'categories and numeric ranges may not represent future applicants. '
            'Do not use this dashboard for lending decisions.</p>'
            '</div>', unsafe_allow_html=True
        )

else:
    page_title("Anomaly Analysis", "DBSCAN exploration on 5,000 training borrowers • eps=1.4, min_samples=10")
    a,b,c,e=st.columns(4)
    a.metric("Training sample","5,000")
    b.metric("Noise observations","809")
    c.metric("Noise share","16.18%")
    e.metric("Default-rate test p-value","0.1806")
    main,side=st.columns([3.2,1.15],gap="large")
    with main:
        a,b=st.columns(2)
        for col,metric,title in [(a,"Historical default rate (%)","Historical default rate"),(b,"Average FICO","Average FICO")]:
            with col:
                fig=px.bar(ANOMALY,x="Group",y=metric,text=metric,title=title,color="Group",color_discrete_map={"Clustered":YELLOW,"Noise":DARK_YELLOW})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(plot_style(fig),use_container_width=True)
        a,b=st.columns(2)
        for col,metric,title in [(a,"Average income","Average original income"),(b,"Average DTI","Average debt-to-income ratio")]:
            with col:
                fig=px.bar(ANOMALY,x="Group",y=metric,text=metric,title=title,color="Group",color_discrete_map={"Clustered":YELLOW,"Noise":DARK_YELLOW})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(plot_style(fig),use_container_width=True)
        st.dataframe(ANOMALY,hide_index=True,use_container_width=True)
    with side:
        st.markdown('''<div class="right-panel"><h4>Objective</h4><p>Identify unusual feature combinations and compare their historical financial characteristics with those of borrowers in dense groups.</p></div>''',unsafe_allow_html=True)
        st.markdown('''<div class="right-panel"><h4>Key takeaways</h4><ul><li>Noise default rate: <b>17.06%</b>.</li><li>Clustered default rate: <b>19.06%</b>.</li><li>Difference: <b>−2.01 percentage points</b>.</li><li>Two-proportion test p-value: <b>0.1806</b>.</li><li>95% CI: <b>−4.73 to +0.97 percentage points</b>.</li></ul></div>''',unsafe_allow_html=True)
        st.markdown('''<div class="right-panel"><h4>Important caution</h4><p>The default-rate difference is not statistically significant at 5%. DBSCAN noise is not a fraud flag or proof of elevated credit risk.</p></div>''',unsafe_allow_html=True)

st.divider()
st.caption("Research dashboard • Historical descriptive analysis only • No individual underwriting decisions • Loan-vintage outcome periods may differ")
