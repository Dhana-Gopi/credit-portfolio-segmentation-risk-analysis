from pathlib import Path
import warnings
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT  # GitHub repository stores CSVs in its root directory
YELLOW = '#F4B400'
GOLD = '#C78B00'
INK = '#202735'
GRAY = '#657080'
PALETTE = ['#F4B400','#E1A200','#F7CF5C','#D68C00','#FFE49B']

st.set_page_config(page_title='Credit Portfolio Intelligence', page_icon='🟡', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
:root {color-scheme: light;}
.stApp {background:#FFFFFF;color:#222222}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #EDE8D9}
.block-container {padding-top:1.5rem;max-width:1480px;padding-left:2rem;padding-right:2rem;padding-bottom:3rem}
h1,h2,h3 {color:#202735 !important;letter-spacing:-.025em}
h1 {font-size:2rem !important}
[data-testid="stMetric"] {background:#fff;border:1px solid #EDE8D9;border-top:3px solid #F4B400;border-radius:13px;padding:15px 17px;box-shadow:0 3px 12px #17171708}
[data-testid="stMetricLabel"] {color:#657080}
[data-testid="stMetricValue"] {color:#202735;font-weight:750}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#EDE8D9 !important}
.stButton>button[kind="primary"],button[kind="primary"] {background:#F4B400;color:#202735;border:0;font-weight:700}
.stButton>button[kind="primary"]:hover {background:#E5A500;color:#202735}
.stTabs [data-baseweb="tab-highlight"] {background:#F4B400}
[data-testid="stDataFrame"] {border:1px solid #EDE8D9;border-radius:10px}
.note {background:#FFFBE6;border-left:4px solid #F4B400;padding:13px 17px;border-radius:8px;margin:12px 0 18px}
.eyebrow {color:#9A7400;text-transform:uppercase;letter-spacing:.12em;font-size:.73rem;font-weight:800}
hr {border-color:#EDE8D9 !important}
.insight-card{background:#FFFBE6;border-left:4px solid #F2C300;border-radius:7px;padding:10px 12px;margin:6px 0 11px;font-size:12px;line-height:1.5;color:#222}
.section-title{font-weight:850;font-size:1.15rem;border-bottom:2px solid #F2C300;padding-bottom:4px;margin:14px 0 9px}
[data-testid="stMetric"]{background:#FFFBE6 !important;border-top:0 !important;border-left:4px solid #F2C300 !important}
[data-testid="stRadio"] [role="radiogroup"]{background:#FFFBE6;border:1px solid #F2C300;border-radius:10px;padding:6px 9px;gap:14px;flex-wrap:wrap}
[data-testid="stRadio"] label{font-weight:700}
.chart-explainer{border-top:1px solid #F1E9D0;padding:11px 4px 2px;color:#596273;font-size:12px;line-height:1.55;min-height:66px}
.chart-explainer b{color:#8D6A00;display:block;font-size:11px;text-transform:uppercase;letter-spacing:.045em;margin-bottom:3px}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:15px !important;box-shadow:0 4px 22px rgba(32,39,53,.045)}
[data-testid="stHorizontalBlock"]{row-gap:1.4rem}
[data-testid="stRadio"] [role="radiogroup"]{justify-content:center;gap:6px !important;background:#fff !important;border:0 !important;border-bottom:1px solid #EDE8D9 !important;border-radius:0 !important;padding-bottom:12px !important}
[data-testid="stRadio"] label{padding:9px 12px;border-radius:9px;white-space:nowrap}
[data-testid="stRadio"] label:has(input:checked){background:#FFF4C2;color:#202735}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_scorecard(path,mtime):
    return pd.read_csv(path)

score_path = DATA/'credit_segment_scorecard.csv'
if not score_path.exists():
    st.error('Missing credit_segment_scorecard.csv in the repository root')
    st.stop()
df=load_scorecard(str(score_path),score_path.stat().st_mtime)
required={'Cluster','Segment_Name','Borrowers','Defaults','Default_Rate_%','Relative_Default_Index','Total_Original_Principal','Principal_Share_%','Default_Share_%','Borrower_Share_%','Avg_Original_Loan'}
if not required.issubset(df.columns):
    st.error('Scorecard is missing required columns: '+', '.join(sorted(required-set(df.columns))))
    st.stop()
for col in required-{'Segment_Name'}:
    df[col]=pd.to_numeric(df[col],errors='raise')
df=df.sort_values('Cluster').reset_index(drop=True)
N=int(df.Borrowers.sum());D=int(df.Defaults.sum());R=100*D/N;P=float(df.Total_Original_Principal.sum())

BENCHMARK=pd.DataFrame({'Algorithm':['K-Means','Hierarchical','DBSCAN'],'Silhouette':[.1418,.0839,.0500],'Clusters':[5,5,9],'Noise share (%)':[0,0,16.18],'Fit time (s)':[.0778,.2148,.0332]})
ELBOW=pd.DataFrame({'K':list(range(2,9)),'Inertia':[257774.81,230326.12,208976.43,192071.65,181523.24,173178.17,167138.15]})
SIL=pd.DataFrame({'K':[4,5,6],'Silhouette':[.1449,.1393,.1364]})
TEMP=pd.DataFrame({'Cluster':[0,1,2,3,4],'Training':[8.29,18.28,19.66,24.80,16.35],'Validation':[10.22,23.61,24.68,31.14,19.85],'Test':[10.45,22.40,24.75,28.99,17.52]})
STABILITY=pd.DataFrame({'K':[4,5],'Across initializations (ARI)':[.9768,.9918],'Across borrower samples (ARI)':[.8826,.8477]})
PROFILES=pd.DataFrame({'Cluster':[0,1,2,3,4],'Average income':[76410,55750,126703,49056,76342],'Average DTI':[13.96,14.77,17.80,28.19,13.65],'Average FICO':[760.96,686.27,699.78,688.88,684.64],'Average loan':[13733,9981,27014,11588,11559],'Average employment years':[None,2.26,None,None,8.98]})
PROFILES=PROFILES.merge(df[['Cluster','Segment_Name']],on='Cluster')

def fig_style(fig,height=390,percent=False):
    fig.update_layout(template='plotly_white',height=height,paper_bgcolor='#fff',plot_bgcolor='#fff',font=dict(family='Arial',color=INK),margin=dict(l=15,r=20,t=40,b=40),legend=dict(orientation='h',y=-.25),hoverlabel=dict(bgcolor='white',font_color=INK))
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor='#E8E8E8')
    fig.update_yaxes(gridcolor='#F0F0F0',zeroline=False)
    if percent:fig.update_yaxes(ticksuffix='%')
    return fig

CHART_MEANINGS = {
    'Borrowers by segment': 'Shows how the historical test borrowers are distributed among the five segments; volume alone does not indicate risk.',
    'Historical default rate by segment': 'Compares the observed frequency of default across segments. The dashed line marks the full test-portfolio average.',
    'Share of original principal': 'Shows where the original loan principal is concentrated; it is not the current outstanding balance or realized loss.',
    'Share of recorded defaults': 'Shows each segment’s contribution to total observed defaults, which depends on both segment size and default rate.',
    'Historical default-rate comparison': 'Compares observed default frequency between selected historical segments; it is not a borrower-level probability.',
    'Principal concentration vs default frequency': 'Each bubble represents a segment: rightward means greater principal share, upward means higher default frequency, and size reflects borrower count.',
    'Contribution to observed defaults': 'Shows what percentage of all observed defaulted borrowers came from each segment.',
    'Relative default index': 'An index above 1.00 means the segment defaulted more frequently than the overall test cohort; below 1.00 means less frequently.',
    'Elbow: K = 2 to 8': 'Shows within-cluster inertia falling as K increases. The bend helps assess diminishing returns; it does not identify a uniquely correct K.',
    'Silhouette: K = 4, 5, 6': 'Shows how distinct the K-Means groups are. Scores near zero indicate overlapping clusters; K=4 is slightly higher here.',
    'Silhouette on comparison sample': 'Compares geometric separation across algorithms on the notebook experiment. DBSCAN excludes noise from its silhouette calculation.',
    'Measured single-run fit time': 'Shows execution time for the sampled experiment only; these are not production performance benchmarks.',
    'K-Means stability (Adjusted Rand Index)': 'Compares consistency across random initializations and across different borrower samples. Higher ARI indicates more similar assignments.',
    'Cluster default rates across training, validation and test': 'Tracks observed segment default rates across historical cohorts; different loan vintages may have different outcome maturity.',
}

def chart(fig,height=340):
    title = str(fig.layout.title.text or '').strip()
    fig=fig_style(fig,height)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15,color=INK),x=0,xanchor='left'),
        margin=dict(l=20,r=34,t=62,b=65),
        uniformtext_minsize=10,uniformtext_mode='hide',
        legend=dict(orientation='h',y=-0.24,x=0,font=dict(size=10)),
        hovermode='closest',
    )
    fig.update_traces(cliponaxis=False,selector=dict(type='bar'))
    with st.container(border=True):
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        meaning=CHART_MEANINGS.get(title)
        if meaning:
            st.markdown(f'<div class="chart-explainer"><b>What this chart denotes</b><div>{meaning}</div></div>',unsafe_allow_html=True)

def note(s):st.markdown(f'<div class="note">{s}</div>',unsafe_allow_html=True)
def heading(title,subtitle):st.markdown('<div class="eyebrow">Credit portfolio intelligence</div>',unsafe_allow_html=True);st.title(title);st.caption(subtitle)
def metric_row(items):
    cols=st.columns(len(items))
    for c,(name,value,helptext) in zip(cols,items):
        c.metric(name,value,help=helptext)

st.markdown('<h1 style="text-align:center;font-weight:900;color:#222;margin-bottom:0">CREDIT PORTFOLIO INTELLIGENCE</h1>',unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666;margin:2px 0 12px">Borrower segmentation • Credit risk patterns • Portfolio exposure • Machine learning</p>',unsafe_allow_html=True)
PAGES=['Dataset Executive Overview','Portfolio Overview','Segment Explorer','Risk & Exposure','Model Lab','Borrower Sandbox']
page=st.radio('Dashboard navigation',PAGES,horizontal=True,label_visibility='collapsed',key='top_navigation')
st.divider()

@st.cache_data(show_spinner=False)
def read_eda(path,mtime):
    return pd.read_csv(path,dtype={'Year':str,'Category':str})
eda_path=DATA/'eda_aggregates.csv'
eda=read_eda(str(eda_path),eda_path.stat().st_mtime) if eda_path.exists() else pd.DataFrame()

def insight(text):
    st.markdown(f'<div class="insight-card">{text}</div>',unsafe_allow_html=True)

def insights_panel(title, points, subtitle=None):
    """Reusable right-hand executive commentary rail."""
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Executive interpretation</div>',unsafe_allow_html=True)
        st.subheader(title)
        if subtitle: st.caption(subtitle)
        for heading_text,body in points:
            st.markdown(
                '<div class="insight-card"><strong>'+escape(str(heading_text))+'</strong><div style="margin-top:5px">'+escape(str(body))+'</div></div>',
                unsafe_allow_html=True)


def filtered_eda(year, dimension):
    if eda.empty: return pd.DataFrame()
    return eda.loc[(eda['Year'].astype(str)==str(year)) & (eda['Dimension']==dimension)].copy()


def eda_visual(frame, metric, title, ordered=False, orientation='v', height=330):
    if frame.empty:
        st.info('No aggregated data available for this selection.')
        return
    frame=frame.copy()
    if ordered: frame=frame.sort_values(metric,ascending=False)
    if orientation=='h':
        fig=px.bar(frame,x=metric,y='Category',orientation='h',color_discrete_sequence=[YELLOW],title=title,
                   hover_data=['Borrowers','Defaults','Default_Rate_%'])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig=px.bar(frame,x='Category',y=metric,color_discrete_sequence=[YELLOW],title=title,
                   hover_data=['Borrowers','Defaults','Default_Rate_%'])
        fig.update_xaxes(tickangle=-20)
    if metric=='Default_Rate_%':fig.update_yaxes(ticksuffix='%' if orientation=='v' else None)
    chart(fig,height)

def eda_chart(data,dimension,kind='count',top=12,height=235):
    x=data[data.Dimension==dimension].copy()
    if x.empty:
        st.info('No data for this view');return x
    if dimension in ['Purpose','State','Employment']:
        x=x.nlargest(top,'Borrowers')
    if kind=='rate':
        fig=px.bar(x,x='Category',y='Default_Rate_%',color_discrete_sequence=[YELLOW],text='Default_Rate_%')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        fig.update_yaxes(ticksuffix='%')
    else:
        fig=px.bar(x,x='Category',y='Borrowers',color_discrete_sequence=[YELLOW],text='Borrowers')
        fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
    fig.update_layout(height=height,margin=dict(l=5,r=5,t=10,b=35),showlegend=False,font=dict(size=10,color='#222'),paper_bgcolor='white',plot_bgcolor='white')
    fig.update_xaxes(title='',tickangle=-25,showgrid=False)
    fig.update_yaxes(title='',gridcolor='#F4F4F4')
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    return x

if page=='Portfolio Overview':
    heading('Portfolio Overview','A concise view of portfolio composition, observed defaults and original principal concentration')
    metric_row([('Test borrowers',f'{N:,}','2017–2018 historical test portfolio'),('Historical defaults',f'{D:,}','Observed default outcomes'),('Observed default rate',f'{R:.2f}%','Historical cohort average'),('Original principal',f'{P/1e9:.2f}B','Source dataset units; not current outstanding exposure')])
    visual,rail=st.columns([3.2,1.15],gap='large')
    with visual:
        a,b=st.columns(2,gap='medium')
        with a:
            fig=px.bar(df.sort_values('Borrowers'),x='Borrowers',y='Segment_Name',orientation='h',text='Borrowers',title='Borrowers by segment',color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate='%{text:,}',textposition='outside');chart(fig)
        with b:
            fig=px.bar(df.sort_values('Default_Rate_%'),x='Default_Rate_%',y='Segment_Name',orientation='h',text='Default_Rate_%',title='Historical default rate by segment',color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside');fig.add_vline(x=R,line_dash='dash',line_color=INK);chart(fig)
        a,b=st.columns(2,gap='medium')
        with a:
            fig=px.bar(df.sort_values('Principal_Share_%'),x='Principal_Share_%',y='Segment_Name',orientation='h',text='Principal_Share_%',title='Share of original principal',color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside');chart(fig)
        with b:
            fig=px.bar(df.sort_values('Default_Share_%'),x='Default_Share_%',y='Segment_Name',orientation='h',text='Default_Share_%',title='Share of recorded defaults',color_discrete_sequence=[YELLOW])
            fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside');chart(fig)
    highest=df.loc[df['Default_Rate_%'].idxmax()]
    largest=df.loc[df['Principal_Share_%'].idxmax()]
    with rail:
        insights_panel('Portfolio key takeaways',[
            ('Highest historical default frequency',f'{highest.Segment_Name}: {highest["Default_Rate_%"]:.2f}% observed default rate. Review debt burden and cohort composition.'),
            ('Largest original principal concentration',f'{largest.Segment_Name}: {largest["Principal_Share_%"]:.2f}% of original principal. This is not current outstanding exposure.'),
            ('Default contribution','Default share measures contribution to total observed default cases, not loss amounts.'),
            ('Interpretation boundary','These are historical test-cohort segment summaries, not borrower-level default predictions.')
        ],'Test cohort only · five K-Means segments')

elif page=='Segment Explorer':
    heading('Segment Explorer','Compare two segments directly, then examine their original financial characteristics')
    a,b=st.columns(2,gap='large')
    names=df.Segment_Name.tolist()
    with a:one=st.selectbox('First segment',names,index=3)
    with b:two=st.selectbox('Compare against',names,index=0)
    sel=df[df.Segment_Name.isin([one,two])].copy()
    fig=px.bar(sel,x='Segment_Name',y='Default_Rate_%',color='Segment_Name',text='Default_Rate_%',color_discrete_sequence=PALETTE,title='Historical default-rate comparison')
    fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside');fig.update_layout(showlegend=False)
    chart(fig)
    col1,col2=st.columns(2,gap='large')
    for col,name in [(col1,one),(col2,two)]:
        s=df[df.Segment_Name==name].iloc[0]
        with col:
            with st.container(border=True):
                st.subheader(name)
                st.metric('Historical default rate',f'{s["Default_Rate_%"]:.2f}%')
                st.metric('Relative default index',f'{s["Relative_Default_Index"]:.2f}×')
                st.metric('Borrowers',f'{int(s.Borrowers):,}')
                st.metric('Principal share',f'{s["Principal_Share_%"]:.2f}%')
    st.subheader('Financial characteristics (training sample)')
    p=PROFILES[PROFILES.Segment_Name.isin([one,two])].set_index('Segment_Name')
    st.dataframe(p[['Average income','Average DTI','Average FICO','Average loan','Average employment years']].T, use_container_width=True)
    st.caption('Profile averages come from the 50,000-borrower training sample; default-rate KPIs come from the later test portfolio. They are different populations.')

elif page=='Risk & Exposure':
    heading('Risk & Exposure','Understand default frequency, concentration and original loan principal separately')
    metric_row([('Historical default rate',f'{R:.2f}%','Observed test portfolio rate'),('Original principal',f'{P/1e9:.2f}B','Original principal in dataset units'),('Highest default segment',f'{high_id if (high_id:=int(df.loc[df["Default_Rate_%"].idxmax(),"Cluster"])) is not None else ""}','Cluster ID'),('Largest principal segment',f'{int(df.loc[df["Principal_Share_%"].idxmax(),"Cluster"])}','Cluster ID')])
    a,b=st.columns(2,gap='large')
    with a:
        fig=px.scatter(df,x='Principal_Share_%',y='Default_Rate_%',size='Borrowers',color='Segment_Name',text='Cluster',color_discrete_sequence=PALETTE,title='Principal concentration vs default frequency',hover_data=['Borrowers','Default_Share_%'])
        fig.update_traces(textposition='top center',marker=dict(line=dict(width=1,color='#8F7900')))
        chart(fig,450)
    with b:
        fig=px.bar(df.sort_values('Default_Share_%'),x='Default_Share_%',y='Segment_Name',orientation='h',color_discrete_sequence=[YELLOW],text='Default_Share_%',title='Contribution to observed defaults')
        fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside')
        chart(fig,450)
    st.subheader('Relative default index')
    fig=px.bar(df,x='Segment_Name',y='Relative_Default_Index',text='Relative_Default_Index',color_discrete_sequence=[YELLOW],title='Relative default index')
    fig.add_hline(y=1,line_dash='dash',line_color='#465468',annotation_text='Portfolio average')
    fig.update_traces(texttemplate='%{text:.2f}×',textposition='outside');fig.update_xaxes(tickangle=-15)
    chart(fig)
    _,risk_rail=st.columns([3.2,1.15],gap='large')
    with risk_rail: insights_panel('Risk interpretation',[('Default frequency','A higher default rate does not necessarily mean larger financial losses.'),('Original principal','Original loan amounts do not represent current exposure, recovery or loss severity.')])

elif page=='Model Lab':
    heading('Model Lab','Saved notebook experiments · comparisons are descriptive, not live retraining')
    metric_row([('Primary algorithm','K-Means','Chosen for interpretability'),('Selected clusters','5','K selected after multiple diagnostics'),('Comparison silhouette','0.1418','5,000-borrower common sample'),('Features','24','Five scaled numerical, one flag, eighteen categorical')])
    tab1,tab2,tab3=st.tabs(['K selection','Algorithm comparison','Stability & visualization'])
    with tab1:
        a,b=st.columns(2,gap='large')
        with a:
            fig=px.line(ELBOW,x='K',y='Inertia',markers=True,title='Elbow: K = 2 to 8',color_discrete_sequence=[GOLD]);fig.add_vline(x=5,line_dash='dash',line_color=INK)
            chart(fig)
        with b:
            fig=px.bar(SIL,x='K',y='Silhouette',text='Silhouette',title='Silhouette: K = 4, 5, 6',color_discrete_sequence=[YELLOW]);fig.update_traces(texttemplate='%{text:.4f}',textposition='outside');chart(fig)
        note('K = 4 has a slightly higher silhouette (0.1449 vs 0.1393). K = 5 was preferred for more interpretable borrower profiles, not because it wins every metric.')
    with tab2:
        a,b=st.columns(2,gap='large')
        with a:
            fig=px.bar(BENCHMARK,x='Algorithm',y='Silhouette',text='Silhouette',color_discrete_sequence=[YELLOW],title='Silhouette on comparison sample');fig.update_traces(texttemplate='%{text:.4f}',textposition='outside');chart(fig)
        with b:
            fig=px.bar(BENCHMARK,x='Algorithm',y='Fit time (s)',text='Fit time (s)',color_discrete_sequence=[YELLOW],title='Measured single-run fit time');fig.update_traces(texttemplate='%{text:.4f}s',textposition='outside');chart(fig)
        st.dataframe(BENCHMARK,hide_index=True,use_container_width=True)
        st.caption('DBSCAN silhouette excludes noise observations, so it is not directly comparable. Times are small-sample measurements, not production benchmarks.')
    with tab3:
        a,b=st.columns(2,gap='large')
        with a:
            fig=px.bar(STABILITY.melt('K',var_name='Test',value_name='ARI'),x='K',y='ARI',color='Test',barmode='group',title='K-Means stability (Adjusted Rand Index)',color_discrete_sequence=[YELLOW,'#465468']);chart(fig)
        with b:
            st.markdown('#### Dimensionality reduction')
            st.metric('PCA · seven components','93.52% variance')
            st.metric('PCA · two components','45.52% variance')
            st.metric('PCA + K-Means silhouette','0.1543')
            st.metric('t-SNE KL divergence','1.5981')
        st.info('The uploaded scorecard does not contain point-level PCA, t-SNE or UMAP coordinates. This version does not fabricate scatterplots. Export those coordinates from the notebook to enable accurate interactive projections.')
        st.subheader('Historical risk patterns across time')
        t=TEMP.melt(id_vars='Cluster',var_name='Period',value_name='Default rate (%)')
        fig=px.line(t,x='Period',y='Default rate (%)',color='Cluster',markers=True,color_discrete_sequence=PALETTE,title='Cluster default rates across training, validation and test')
        chart(fig)
        st.caption('Training: sampled pre-2016 borrowers; validation: 2016; test: 2017–2018. Outcome maturity may differ across vintages.')

elif page=='Dataset Executive Overview':
    heading('Dataset Executive Overview','Explore the original loan dataset through historical, aggregated EDA with transparent year and dimension filters')
    if eda.empty:
        st.error('EDA aggregates missing. Place eda_aggregates.csv in the repository root.')
        st.stop()
    years=['All']+sorted([y for y in eda['Year'].astype(str).unique() if y!='All'])
    filters=st.columns([1,1,1.4],gap='large')
    with filters[0]:year=st.selectbox('Loan issue year',years,index=0,key='eda_year')
    with filters[1]:measure=st.selectbox('Chart measure',['Borrowers','Default_Rate_%','Defaults','Principal'],key='eda_measure',format_func=lambda x:{'Borrowers':'Borrower count','Default_Rate_%':'Historical default rate','Defaults':'Default cases','Principal':'Original principal'}[x])
    with filters[2]:focus=st.selectbox('Detailed breakdown',['Income','DTI','FICO','Loan size','Home ownership','Purpose','Employment','State'],key='eda_focus')
    vintage=filtered_eda(year,'Vintage')
    income=filtered_eda(year,'Income');dti_frame=filtered_eda(year,'DTI');fico=filtered_eda(year,'FICO')
    baseline=filtered_eda(year,'Income')
    borrowers=int(baseline.Borrowers.sum()) if not baseline.empty else 0
    defaults=int(baseline.Defaults.sum()) if not baseline.empty else 0
    principal=float(baseline.Principal.sum()) if not baseline.empty else 0
    rate=100*defaults/borrowers if borrowers else 0
    metric_row([('Borrowers in selected cohort',f'{borrowers:,}','Source dataset; all records for selected year'),
                ('Historical default cases',f'{defaults:,}','Observed defaults in the selected cohort'),
                ('Historical default rate',f'{rate:.2f}%','Default cases divided by borrowers'),
                ('Original principal',f'{principal/1e9:.2f}B','Source currency units, not current balance')])
    st.caption('The year selector filters each precomputed aggregate independently. Category charts are not jointly cross-filtered; for example, choosing an income band does not filter the DTI chart. Aggregates describe the full source dataset, not the K-Means test scorecard.')
    left,right=st.columns([3.2,1.15],gap='large')
    with left:
        st.markdown('### Financial profile at a glance')
        a,b=st.columns(2,gap='medium')
        with a:eda_visual(income,measure,'Income distribution / default patterns')
        with b:eda_visual(dti_frame,measure,'Debt-to-income distribution / defaults')
        a,b=st.columns(2,gap='medium')
        with a:eda_visual(fico,measure,'FICO score bands')
        with b:eda_visual(filtered_eda(year,'Loan size'),measure,'Original loan-size distribution')
        st.markdown('### Income vs debt burden — aggregated comparison')
        comparison=filtered_eda(year,focus)
        if not comparison.empty:
            comparison=comparison.sort_values('Borrowers',ascending=False).head(18)
            fig=px.scatter(comparison,x='Avg_Income',y='Avg_DTI',size='Borrowers',color='Default_Rate_%',
                hover_name='Category',hover_data={'Borrowers':':,','Defaults':':,','Default_Rate_%':':.2f','Avg_FICO':':.1f'},
                color_continuous_scale=['#FFE49B','#F4B400','#9C6900'],title=f'Average income vs average DTI · {focus} categories',size_max=54)
            fig.update_xaxes(title='Average annual income (source units)')
            fig.update_yaxes(title='Average debt-to-income ratio')
            chart(fig,420)
        st.markdown('### Borrower composition and risk by category')
        a,b=st.columns(2,gap='medium')
        with a:eda_visual(filtered_eda(year,'Home ownership'),measure,'Home ownership')
        with b:eda_visual(filtered_eda(year,'Purpose').nlargest(10,'Borrowers'),measure,'Top 10 loan purposes',orientation='h')
        st.markdown('### Drill-down table')
        details=filtered_eda(year,focus).sort_values('Borrowers',ascending=False)
        st.dataframe(details[['Category','Borrowers','Defaults','Default_Rate_%','Avg_Income','Avg_FICO','Avg_DTI','Principal']].rename(columns={'Default_Rate_%':'Default rate (%)','Avg_Income':'Avg income','Avg_FICO':'Avg FICO','Avg_DTI':'Avg DTI','Principal':'Original principal'}),hide_index=True,use_container_width=True)
        csv=details.to_csv(index=False).encode('utf-8')
        st.download_button('Download filtered aggregate table',csv,file_name=f'eda_{focus.lower().replace(" ","_")}_{year}.csv',mime='text/csv')
    with right:
        highest_dti=dti_frame.loc[dti_frame['Default_Rate_%'].idxmax()] if not dti_frame.empty else None
        highest_fico=fico.loc[fico['Default_Rate_%'].idxmin()] if not fico.empty else None
        largest_income=income.loc[income['Borrowers'].idxmax()] if not income.empty else None
        messages=[('Selected cohort',f'{borrowers:,} borrowers with {rate:.2f}% historical defaults. These are descriptive source-data figures.'),
                  ('Income mix',f'Largest income band by borrower count: {largest_income.Category} ({int(largest_income.Borrowers):,} borrowers).' if largest_income is not None else 'Income distribution unavailable.'),
                  ('Debt burden',f'Highest observed DTI-band default rate: {highest_dti.Category} ({highest_dti["Default_Rate_%"]:.2f}%).' if highest_dti is not None else 'DTI data unavailable.'),
                  ('Credit-score profile',f'Lowest observed FICO-band default rate: {highest_fico.Category} ({highest_fico["Default_Rate_%"]:.2f}%).' if highest_fico is not None else 'FICO data unavailable.'),
                  ('How to read the bubble chart','Each bubble is a category average, not an individual borrower. Bubble size shows borrower count; colour shows historical default rate.'),
                  ('Business caution','Patterns are associations, not causal explanations. Different issue-year cohorts can have different time available to default.')]
        insights_panel('Executive EDA takeaways',messages,f'Issue year: {year} · Original dataset')
        with st.container(border=True):
            st.markdown('#### Analyst notes')
            st.caption('The displayed measures come from saved EDA aggregates. There is no fabricated borrower-level joint distribution or live model retraining.')
            st.caption('Income, DTI and FICO category averages are calculated independently within each breakdown. A year filter is applied consistently to every view.')

elif page=='Borrower Sandbox':
    heading('Borrower Sandbox','Apply the saved preprocessing and K-Means model to one illustrative borrower')
    st.warning('This is a descriptive segment assignment, not a default probability, credit score, approval decision or lending recommendation.')
    a,b=st.columns([1.8,1],gap='large')
    with a:
        with st.form('borrower'):
            l,r=st.columns(2,gap='large')
            with l:
                income=st.number_input('Annual income (source units)',min_value=1.,value=75000.,step=1000.)
                amount=st.number_input('Original loan amount (source units)',min_value=1.,value=15000.,step=500.)
                fico=st.number_input('FICO score',min_value=300,max_value=850,value=710)
                employment=st.selectbox('Employment length',['< 1 year']+[f'{i} year' if i==1 else f'{i} years' for i in range(1,10)]+['10+ years','NI'],index=5)
            with r:
                dti=st.number_input('Debt-to-income ratio',min_value=0.,value=18.,step=.5)
                ownership=st.selectbox('Home ownership',['MORTGAGE','RENT','OWN','OTHER'])
                purpose=st.selectbox('Loan purpose',['car','credit_card','debt_consolidation','educational','home_improvement','house','major_purchase','medical','moving','other','renewable_energy','small_business','vacation','wedding'],index=2)
            submit=st.form_submit_button('Assign borrower segment',type='primary',use_container_width=True)
        if submit:
            try:
                import joblib
                from borrower_preprocessing import preprocess_borrowers
                bundle=joblib.load(ROOT/'final_kmeans_model.joblib')
                raw=pd.DataFrame([{'revenue':income,'loan_amnt':amount,'fico_n':fico,'dti_n':dti,'emp_length':employment,'home_ownership_n':ownership,'purpose':purpose}])
                features=preprocess_borrowers(raw)
                if features.columns.tolist()!=bundle['feature_columns']:
                    raise ValueError('Saved model and preprocessing feature order differ')
                cluster=int(bundle['model'].predict(features)[0])
                segment=bundle['segment_mapping'][cluster]
                st.session_state['result']=(cluster,segment)
            except Exception as exc:
                st.session_state.pop('result',None)
                st.error(f'Unable to assign segment: {exc}')
        if 'result' in st.session_state:
            cluster,segment=st.session_state['result']
            s=df[df.Cluster==cluster].iloc[0]
            st.success(f'Assigned: Cluster {cluster} — {segment}')
            metric_row([('Segment default rate',f'{s["Default_Rate_%"]:.2f}%','Historical test segment, not individual probability'),('Relative default index',f'{s["Relative_Default_Index"]:.2f}×','Compared with test average'),('Segment borrowers',f'{int(s.Borrowers):,}','Historical test count')])
            st.caption('This assignment describes similarity to historical groups; the displayed default rate does not predict this borrower’s outcome.')
    with b:
        with st.container(border=True):
            st.subheader('What happens behind the form?')
            st.markdown('**1.** Validate raw borrower inputs.\n\n**2.** Log-transform income and cap DTI.\n\n**3.** Convert and impute employment duration.\n\n**4.** Apply saved numerical scaler and categorical encoder.\n\n**5.** Send 24 features to the fitted K-Means model.')
            st.divider()
            st.caption('Source currency and units have not been independently verified. Avoid using this demonstration for underwriting.')

st.divider()
st.caption('Credit Portfolio Intelligence · White & Yellow Edition · Historical descriptive research · No individual underwriting decisions')
