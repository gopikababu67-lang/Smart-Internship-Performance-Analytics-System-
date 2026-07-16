import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Smart Internship Performance Analytics System",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0d7377, #14a085);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    color: white;
    font-size: 1.8rem;
    margin: 0;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
}
.kpi-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0d7377;
}
.kpi-label {
    font-size: 0.78rem;
    color: #666;
    margin-top: 2px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0d7377;
    border-left: 4px solid #0d7377;
    padding-left: 10px;
    margin: 1.2rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

@st.cache_resource
def load_model(model_path, features_path):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def build_model_files(data_path, model_path, features_path):
    df = pd.read_excel(data_path)
    branch_categories = list(pd.unique(df['branch']))
    college_categories = list(pd.unique(df['college_tier']))

    mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    X = pd.DataFrame({
        'age': df['age'].astype(float),
        'gender': df['gender'].map(mapping).fillna(2).astype(int),
        'cgpa': df['cgpa'].astype(float),
        'branch': df['branch'].map({v: i for i, v in enumerate(branch_categories)}).astype(int),
        'college_tier': df['college_tier'].map({v: i for i, v in enumerate(college_categories)}).astype(int),
        'internships': df['internships_count'].astype(float),
        'projects': df['projects_count'].astype(float),
        'certifications': df['certifications_count'].astype(float),
        'coding_score': df['coding_skill_score'].astype(float),
        'aptitude': df['aptitude_score'].astype(float),
        'communication': df['communication_skill_score'].astype(float),
        'hackathons': df['hackathons_participated'].astype(float),
        'github_repos': df['github_repos'].astype(float),
        'mock_interview': df['mock_interview_score'].astype(float),
        'attendance': df['attendance_percentage'].astype(float),
        'backlogs': df['backlogs'].astype(float),
        'overall_skill': df['overall_skill_score'].astype(float),
        'placement_readiness': df['placement_readiness'].astype(float),
    })

    y = df['placed_flag'].astype(int)
    feature_cols = [
        'age', 'gender', 'cgpa', 'branch', 'college_tier',
        'internships', 'projects', 'certifications',
        'coding_score', 'aptitude', 'communication',
        'hackathons', 'github_repos', 'mock_interview',
        'attendance', 'backlogs', 'overall_skill', 'placement_readiness'
    ]

    X = X[feature_cols]
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))
possible_paths = [
    os.path.join(root_dir, "data", "combined_dataset_FINAL_CLEAN.xlsx"),
    os.path.join(root_dir, "combined_dataset_FINAL_CLEAN.xlsx"),
    os.path.join(base_dir, "data", "combined_dataset_FINAL_CLEAN.xlsx"),
    os.path.join(base_dir, "combined_dataset_FINAL_CLEAN.xlsx"),
]

data_path = next((p for p in possible_paths if os.path.exists(p)), None)
if data_path is None:
    st.error(
        "Cannot find `combined_dataset_FINAL_CLEAN.xlsx`.\n"
        "Place the file in the project root or in the `data/` folder next to `app.py`."
    )
    st.stop()

model_path = os.path.join(root_dir, "models", "placement_model.pkl")
features_path = os.path.join(root_dir, "models", "feature_columns.pkl")
if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.info("Model artifacts are missing. Training a new model now...")
    try:
        build_model_files(data_path, model_path, features_path)
        st.success("Model files generated successfully. Restart the app if necessary.")
    except Exception as exc:
        st.error(
            "Unable to generate model files automatically.\n"
            "Please run `python train_model.py` locally or place the generated files in `models/`.\n"
            f"Error: {exc}"
        )
        st.stop()


df = load_data(data_path)
model, feature_cols = load_model(model_path, features_path)

# ── Sidebar Navigation ───────────────────────────────────
st.sidebar.markdown("## 🎓 SIPAS Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview Dashboard",
    "📊 EDA Analysis",
    "🤖 ML Prediction",
    "📈 Skill Gap Analysis",
    "💡 Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
branch_filter = st.sidebar.multiselect(
    "Branch", options=df['branch'].unique(), default=df['branch'].unique()
)
tier_filter = st.sidebar.multiselect(
    "College Tier", options=df['college_tier'].unique(), default=df['college_tier'].unique()
)
gender_filter = st.sidebar.multiselect(
    "Gender", options=df['gender'].unique(), default=df['gender'].unique()
)

# Apply filters
dff = df[
    df['branch'].isin(branch_filter) &
    df['college_tier'].isin(tier_filter) &
    df['gender'].isin(gender_filter)
]

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 Smart Internship Performance Analytics System</h1>
  <p>Data Analytics | Python | Power BI | Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════
if page == "🏠 Overview Dashboard":
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    total = len(dff)
    placed = dff['placed_flag'].sum()
    placement_rate = (placed / total * 100) if total > 0 else 0
    avg_cgpa = dff['cgpa'].mean()
    avg_skill = dff['overall_skill_score'].mean()
    avg_salary = dff[dff['salary_package_lpa'] > 0]['salary_package_lpa'].mean()

    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total/1000:.0f}K</div><div class="kpi-label">Total Students</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_cgpa:.2f}</div><div class="kpi-label">Average CGPA</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_skill:.1f}</div><div class="kpi-label">Avg Skill Score</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{dff["internships_count"].sum()/1000:.0f}K</div><div class="kpi-label">Total Internships</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{placement_rate:.1f}%</div><div class="kpi-label">Placement Rate</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_salary:.1f}</div><div class="kpi-label">Avg Salary LPA</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Placement Status Distribution</div>', unsafe_allow_html=True)
        place_counts = dff['placement_status'].value_counts()
        fig = px.pie(values=place_counts.values, names=place_counts.index,
                     color_discrete_sequence=['#0d7377','#aaaaaa'],
                     hole=0.5)
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Branch-wise Student Count</div>', unsafe_allow_html=True)
        branch_counts = dff['branch'].value_counts().reset_index()
        branch_counts.columns = ['branch','count']
        fig = px.bar(branch_counts, x='count', y='branch', orientation='h',
                     color='count', color_continuous_scale='teal')
        fig.update_layout(height=300, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Placement by Gender</div>', unsafe_allow_html=True)
        gender_place = dff.groupby(['gender','placement_status']).size().reset_index(name='count')
        gender_place['gender'] = gender_place['gender'].astype(str)
        gender_place['placement_status'] = gender_place['placement_status'].astype(str)

        status_order = [status for status in ['Not Placed', 'Placed'] if status in gender_place['placement_status'].unique()]
        if not status_order:
            status_order = list(gender_place['placement_status'].unique())

        fig = go.Figure()
        colors = ['#aaaaaa', '#0d7377']
        for idx, status in enumerate(status_order):
            subset = gender_place[gender_place['placement_status'] == status]
            fig.add_trace(go.Bar(
                x=subset['gender'],
                y=subset['count'],
                name=status,
                marker_color=colors[idx % len(colors)]
            ))

        fig.update_layout(height=300, margin=dict(t=10, b=10), barmode='group')
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">CGPA Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(dff, x='cgpa', nbins=30,
                           color_discrete_sequence=['#0d7377'])
        fig.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

# ════════════════════════════════════════════════════════
# PAGE 2 — EDA ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📊 EDA Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Skill Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(dff, x='overall_skill_score', nbins=40,
                           color_discrete_sequence=['#14a085'])
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Attendance Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(dff, x='attendance_percentage', nbins=30,
                           color_discrete_sequence=['#0d7377'])
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">CGPA vs Placement Readiness</div>', unsafe_allow_html=True)
        required_cols = ['cgpa', 'placement_readiness', 'placement_status']
        if not all(col in dff.columns for col in required_cols):
            st.warning("Required columns for CGPA vs Placement Readiness visualization are missing.")
        else:
            scatter_data = dff.loc[:, required_cols].dropna().copy()
            if pd.api.types.is_categorical_dtype(scatter_data['placement_status']):
                scatter_data['placement_status'] = scatter_data['placement_status'].cat.remove_unused_categories()
            scatter_data['placement_status'] = scatter_data['placement_status'].astype(str)
            scatter_data = scatter_data.sample(min(2000, len(scatter_data))) if len(scatter_data) > 0 else scatter_data

            if scatter_data.empty:
                st.info("No records available for CGPA vs Placement Readiness.")
            else:
                if pd.api.types.is_categorical_dtype(scatter_data['placement_status']):
                    scatter_data['placement_status'] = scatter_data['placement_status'].cat.remove_unused_categories()
                scatter_data['placement_status'] = scatter_data['placement_status'].astype(str)
                scatter_data = scatter_data.sample(min(2000, len(scatter_data)))

                fig = go.Figure()
                colors = {'Not Placed': '#aaaaaa', 'Placed': '#0d7377'}
                for status in scatter_data['placement_status'].unique():
                    subset = scatter_data[scatter_data['placement_status'] == status]
                    if subset.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=subset['cgpa'],
                        y=subset['placement_readiness'],
                        mode='markers',
                        name=status,
                        marker=dict(color=colors.get(status, '#0d7377'), opacity=0.5, size=6),
                        hovertemplate='CGPA: %{x}<br>Readiness: %{y}<br>Status: ' + status
                    ))

                fig.update_layout(height=280, margin=dict(t=10, b=10), xaxis_title='CGPA', yaxis_title='Placement Readiness')
                st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Internships vs Placement Rate</div>', unsafe_allow_html=True)
        intern_place = dff.groupby('internships_count')['placed_flag'].mean().reset_index()
        intern_place['placement_rate'] = intern_place['placed_flag'] * 100
        fig = px.bar(intern_place, x='internships_count', y='placement_rate',
                     color='placement_rate', color_continuous_scale='teal')
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">College Tier vs Placement Rate</div>', unsafe_allow_html=True)
        tier_place = dff.groupby('college_tier')['placed_flag'].mean().reset_index()
        tier_place['placement_rate'] = tier_place['placed_flag'] * 100
        fig = px.bar(tier_place, x='college_tier', y='placement_rate',
                     color='placement_rate', color_continuous_scale='teal')
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Backlogs Impact on Placement</div>', unsafe_allow_html=True)
        backlog_place = dff.groupby('backlogs')['placed_flag'].mean().reset_index()
        backlog_place['placement_rate'] = backlog_place['placed_flag'] * 100
        fig = px.line(backlog_place, x='backlogs', y='placement_rate',
                      markers=True, color_discrete_sequence=['#0d7377'])
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, width='stretch')

# ════════════════════════════════════════════════════════
# PAGE 3 — ML PREDICTION
# ════════════════════════════════════════════════════════
elif page == "🤖 ML Prediction":
    st.markdown('<div class="section-title">Placement Prediction — Enter Student Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 30, 21)
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
        internships = st.slider("Internships Count", 0, 10, 2)
        projects = st.slider("Projects Count", 0, 15, 4)
        certifications = st.slider("Certifications", 0, 10, 3)
        backlogs = st.slider("Backlogs", 0, 10, 0)

    with col2:
        coding_score = st.slider("Coding Skill Score", 0.0, 100.0, 70.0)
        aptitude = st.slider("Aptitude Score", 0.0, 100.0, 65.0)
        communication = st.slider("Communication Score", 0.0, 100.0, 60.0)
        mock_interview = st.slider("Mock Interview Score", 0.0, 100.0, 65.0)
        attendance = st.slider("Attendance %", 0.0, 100.0, 80.0)
        hackathons = st.slider("Hackathons", 0, 10, 2)

    with col3:
        github_repos = st.slider("GitHub Repos", 0, 50, 5)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        branch = st.selectbox("Branch", df['branch'].unique())
        college_tier = st.selectbox("College Tier", df['college_tier'].unique())

    overall_skill = (coding_score + aptitude + communication + mock_interview) / 4
    placement_readiness = (cgpa * 10 + overall_skill) / 2

    if st.button("🔮 Predict Placement", width='stretch'):
        input_data = pd.DataFrame([[
            age,
            {"Male":0,"Female":1,"Other":2}.get(gender, 0),
            cgpa,
            list(df['branch'].unique()).index(branch) if branch in df['branch'].unique() else 0,
            list(df['college_tier'].unique()).index(college_tier) if college_tier in df['college_tier'].unique() else 0,
            internships, projects, certifications,
            coding_score, aptitude, communication,
            hackathons, github_repos, mock_interview,
            attendance, backlogs, overall_skill, placement_readiness
        ]], columns=feature_cols)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if prediction == 1:
                st.success("✅ LIKELY TO BE PLACED!")
            else:
                st.error("❌ NEEDS IMPROVEMENT")
        with res_col2:
            st.metric("Placement Probability", f"{probability:.1f}%")
        with res_col3:
            st.metric("Placement Readiness Score", f"{placement_readiness:.1f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Placement Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#0d7377"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 70], 'color': "#fff3cc"},
                    {'range': [70, 100], 'color': "#ccf5e7"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

# ════════════════════════════════════════════════════════
# PAGE 4 — SKILL GAP ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📈 Skill Gap Analysis":
    st.markdown('<div class="section-title">Skill Gap Analysis</div>', unsafe_allow_html=True)

    st.markdown("Enter your scores to see your skill gap vs industry average:")
    col1, col2 = st.columns(2)
    with col1:
        s_coding = st.slider("Your Coding Score", 0.0, 100.0, 60.0)
        s_aptitude = st.slider("Your Aptitude Score", 0.0, 100.0, 55.0)
        s_comm = st.slider("Your Communication Score", 0.0, 100.0, 58.0)
    with col2:
        s_mock = st.slider("Your Mock Interview Score", 0.0, 100.0, 55.0)
        s_github = st.slider("Your GitHub Repos", 0, 50, 5)
        s_projects = st.slider("Your Projects Count", 0, 15, 3)

    industry_avg = {
        'Coding': df['coding_skill_score'].mean(),
        'Aptitude': df['aptitude_score'].mean(),
        'Communication': df['communication_skill_score'].mean(),
        'Mock Interview': df['mock_interview_score'].mean(),
        'GitHub (x2)': df['github_repos'].mean() * 2,
        'Projects (x5)': df['projects_count'].mean() * 5
    }
    student_scores = {
        'Coding': s_coding,
        'Aptitude': s_aptitude,
        'Communication': s_comm,
        'Mock Interview': s_mock,
        'GitHub (x2)': s_github * 2,
        'Projects (x5)': s_projects * 5
    }

    categories = list(industry_avg.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(student_scores.values()),
        theta=categories, fill='toself',
        name='You', line_color='#0d7377'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(industry_avg.values()),
        theta=categories, fill='toself',
        name='Industry Avg', line_color='#e74c3c',
        opacity=0.5
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400, title="Your Skills vs Industry Average"
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Your Skill Gaps</div>', unsafe_allow_html=True)
    for skill in categories:
        gap = industry_avg[skill] - student_scores[skill]
        if gap > 5:
            st.warning(f"⚠️ **{skill}**: You are {gap:.1f} points below industry average — needs improvement!")
        else:
            st.success(f"✅ **{skill}**: You are at or above industry average — great!")

# ════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════
elif page == "💡 Recommendations":
    st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)

    your_cgpa = st.slider("Your CGPA", 5.0, 10.0, 7.5, 0.1)
    your_internships = st.slider("Internships completed", 0, 10, 1)
    your_certifications = st.slider("Certifications", 0, 10, 2)
    your_backlogs = st.slider("Backlogs", 0, 10, 0)

    st.markdown("---")
    st.markdown("### 📋 Your Personalized Action Plan")

    if your_cgpa < 7.0:
        st.error("📚 **CGPA Alert**: Focus on academics — aim for 7.5+ CGPA")
    elif your_cgpa < 8.0:
        st.warning("📚 **CGPA**: Good! Try to push above 8.0 for better opportunities")
    else:
        st.success("📚 **CGPA**: Excellent! Maintain your academic performance")

    if your_internships == 0:
        st.error("💼 **Internships**: Apply immediately on Internshala, LinkedIn, LetsIntern")
    elif your_internships == 1:
        st.warning("💼 **Internships**: Good start! Try to complete at least 2 internships")
    else:
        st.success("💼 **Internships**: Great! Your internship experience will help in placements")

    if your_certifications < 3:
        st.warning("🏅 **Certifications**: Complete at least 3 — try Python (NPTEL), SQL (HackerRank), AWS")
    else:
        st.success("🏅 **Certifications**: Great number of certifications!")

    if your_backlogs > 0:
        st.error(f"⚠️ **Backlogs**: Clear your {your_backlogs} backlog(s) — this reduces placement chances by 40%")
    else:
        st.success("✅ **Backlogs**: No backlogs — excellent!")

    st.markdown("---")
    st.markdown("### 🔗 Recommended Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Coding Practice**\n- LeetCode\n- HackerRank\n- CodeChef\n- GeeksforGeeks")
    with col2:
        st.info("**Free Certifications**\n- NPTEL Python\n- AWS Free Tier\n- Google Analytics\n- Microsoft Power BI")
    with col3:
        st.info("**Internship Portals**\n- Internshala\n- LinkedIn\n- LetsIntern\n- Unstop")

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:12px'>"
    "Smart Internship Performance Analytics System | "
    "Built with Python, Streamlit & Machine Learning | "
    "Gopika B — Prathyusha Engineering College</p>",
    unsafe_allow_html=True
)