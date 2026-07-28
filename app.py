import streamlit as st
import pandas as pd
import numpy as np
import re

# Page Configuration
st.set_page_config(
    page_title="DataGrammarly - AI Data Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Grammarly-style Aesthetics and Floating Drawer
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Issue Cards */
    .issue-card {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #ccc;
        background-color: #f8f9fa;
    }
    .issue-critical { border-left-color: #ff4b4b; background-color: #fff5f5; }
    .issue-high { border-left-color: #ffa726; background-color: #fff8e1; }
    .issue-low { border-left-color: #29b6f6; background-color: #e1f5fe; }
    
    .badge {
        font-size: 11px;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
        color: white;
        text-transform: uppercase;
    }
    .badge-critical { background-color: #ff4b4b; }
    .badge-high { background-color: #ffa726; }
    .badge-low { background-color: #29b6f6; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "ignored_issues" not in st.session_state:
    st.session_state.ignored_issues = set()
if "drawer_open" not in st.session_state:
    st.session_state.drawer_open = True

def generate_sample_data():
    """Generates a rich, messy dataset to demonstrate AI inspections."""
    np.random.seed(42)
    data = {
        "Customer ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 101],
        "Full Name": ["Alice Smith", "bob jones ", "CHARLIE BROWN", "  Diana Prince", None, "Evan Wright", "Fiona Gallagher", "George Clark", "Hannah Abbott", "Alice Smith"],
        "Email": ["alice@example.com", "bob.jones@work.org", "charlie#invalid.com", "diana@test.net", "missing@email.com", "evan@domain", "fiona@company.com", "george@mail.org", "hannah@web.io", "alice@example.com"],
        "Age": [28, 34, -5, 45, 29, np.nan, 41, 38, 52, 28],
        "Revenue ($)": [1200.50, 450.00, 3200.00, np.nan, 890.20, 1500.00, 2300.00, 4100.00, 120.00, 1200.50],
        "Signup Date": ["2023-01-15", "2023-02-30", "2023-03-12", "2023/04/05", "Not a date", "2023-05-20", "2023-06-11", "2023-07-01", "2023-08-19", "2023-01-15"]
    }
    return pd.DataFrame(data)

def find_column_by_keyword(df, keywords):
    """Case-insensitive flexible column finder."""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in str(col).strip().lower():
                return col
    return None

def analyze_dataset(df):
    """Performs comprehensive data quality health audit."""
    issues = []
    score = 100

    # 1. Missing Values Check
    missing_count = df.isnull().sum().sum()
    if missing_count > 0 and "missing_vals" not in st.session_state.ignored_issues:
        issues.append({
            "id": "missing_vals",
            "type": "Critical",
            "title": "Missing Values Detected",
            "msg": f"Found **{missing_count} null entries** across columns.",
            "impact": "Can bias analytical models and aggregate calculations.",
            "fix": "impute_missing"
        })
        score -= 20

    # 2. Duplicate Records Check
    dupes = df.duplicated().sum()
    if dupes > 0 and "duplicate_rows" not in st.session_state.ignored_issues:
        issues.append({
            "id": "duplicate_rows",
            "type": "High",
            "title": "Duplicate Rows Found",
            "msg": f"Detected **{dupes} identical duplicate record(s)**.",
            "impact": "Skews metrics and overcounts user records.",
            "fix": "remove_duplicates"
        })
        score -= 20

    # 3. Invalid Negative Values in Age
    age_col = find_column_by_keyword(df, ["age"])
    if age_col:
        neg_count = (pd.to_numeric(df[age_col], errors='coerce') < 0).sum()
        if neg_count > 0 and "negative_age" not in st.session_state.ignored_issues:
            issues.append({
                "id": "negative_age",
                "type": "Critical",
                "title": f"Impossible Values in '{age_col}'",
                "msg": f"Detected **{neg_count} negative age value(s)**.",
                "impact": "Data corruption error in source input.",
                "fix": "fix_negative_age"
            })
            score -= 15

    # 4. Leading/Trailing Whitespace
    name_col = find_column_by_keyword(df, ["name", "customer", "user"])
    if name_col:
        ws_count = df[name_col].dropna().astype(str).apply(lambda x: x != x.strip()).sum()
        if ws_count > 0 and "whitespace_text" not in st.session_state.ignored_issues:
            issues.append({
                "id": "whitespace_text",
                "type": "Low",
                "title": f"Untrimmed Space in '{name_col}'",
                "msg": f"Found **{ws_count} text entries** with leading/trailing spaces.",
                "impact": "Causes database lookup and string matching failures.",
                "fix": "trim_whitespace"
            })
            score -= 10

    # 5. Invalid Email Format Check
    email_col = find_column_by_keyword(df, ["email", "mail"])
    if email_col:
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = df[email_col].dropna().astype(str).apply(lambda x: not bool(re.match(email_regex, x))).sum()
        if invalid_emails > 0 and "invalid_email" not in st.session_state.ignored_issues:
            issues.append({
                "id": "invalid_email",
                "type": "High",
                "title": f"Invalid Emails in '{email_col}'",
                "msg": f"Found **{invalid_emails} malformed email address(es)**.",
                "impact": "Will cause messaging/notification delivery failures.",
                "fix": "flag_invalid_emails"
            })
            score -= 15

    return max(0, score), issues

# Safe Cleaning Operations
def apply_fix_by_id(issue_id):
    df = st.session_state.df
    if issue_id == "remove_duplicates":
        st.session_state.df = df.drop_duplicates()
        st.session_state.history.append("Removed duplicate rows.")
    elif issue_id == "impute_missing":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")
        st.session_state.history.append("Imputed missing values with median/mode.")
    elif issue_id == "negative_age":
        age_col = find_column_by_keyword(df, ["age"])
        if age_col:
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
            df[age_col] = df[age_col].apply(lambda x: abs(x) if pd.notnull(x) else x)
            st.session_state.history.append(f"Converted negative values in '{age_col}' to positive.")
    elif issue_id == "whitespace_text":
        name_col = find_column_by_keyword(df, ["name", "customer", "user"])
        if name_col:
            df[name_col] = df[name_col].astype(str).str.strip()
            st.session_state.history.append(f"Trimmed leading/trailing whitespace in '{name_col}'.")

# Header Navigation Bar
st.title("✨ DataGrammarly Assistant")
st.caption("Proactive automated dataset analysis, cleaning, and quality monitoring.")

# Top Controls Bar
col_load, col_export, col_status = st.columns([2, 2, 2])

with col_load:
    source = st.selectbox("Data Source", ["Select Data Source...", "Use Sample Dataset", "Upload CSV File"], label_visibility="collapsed")
    if source == "Use Sample Dataset" and (st.session_state.df is None or len(st.session_state.history) == 0):
        st.session_state.df = generate_sample_data()
        st.session_state.history = ["Loaded sample dataset."]
        st.rerun()

with col_export:
    if source == "Upload CSV File":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded is not None and st.session_state.df is None:
            st.session_state.df = pd.read_csv(uploaded)
            st.session_state.history = ["Uploaded CSV dataset."]
            st.rerun()

with col_status:
    if st.session_state.df is not None:
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()

# Main Workspace Tabs
if st.session_state.df is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Live Dataset Preview", "📊 Quality Profile", "📜 Action Audit Trail"])
    
    with tab1:
        st.dataframe(st.session_state.df, use_container_width=True, height=450)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", st.session_state.df.shape[0])
        c2.metric("Total Columns", st.session_state.df.shape[1])
        c3.metric("Memory Usage", f"{st.session_state.df.memory_usage().sum() / 1024:.1f} KB")

    with tab2:
        st.write("### Numerical Column Summary")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)

    with tab3:
        st.write("### Audit Log & History")
        if not st.session_state.history:
            st.info("No modifications made yet.")
        else:
            for idx, entry in enumerate(st.session_state.history, 1):
                st.markdown(f"**Step {idx}:** {entry}")

else:
    st.info("👈 Select **'Use Sample Dataset'** or upload a file above to begin analysis.")

# --- Grammarly Right Sidebar / Inspection Drawer ---
with st.sidebar:
    st.markdown("## 💡 Grammarly Insights")
    st.markdown("Automated suggestions for dataset quality improvement.")
    st.divider()

    if st.session_state.df is not None:
        score, issues = analyze_dataset(st.session_state.df)

        # Health Score Gauge Color
        score_color = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
        st.markdown(f"### {score_color} Data Health Score: **{score}/100**")
        st.progress(score / 100)
        
        st.divider()

        if not issues:
            st.success("🎉 **Dataset is Pristine!** No quality issues detected.")
        else:
            st.markdown(f"**{len(issues)} Suggestions Available**")

            # Bulk Actions
            if st.button("⚡ Apply All Suggestions", use_container_width=True, type="primary"):
                for issue in issues:
                    apply_fix_by_id(issue["id"])
                st.session_state.ignored_issues.clear()
                st.rerun()

            st.divider()

            # Render Cards for Each Issue
            for issue in issues:
                badge_class = f"badge-{issue['type'].lower()}"
                
                with st.container():
                    st.markdown(f"""
                    <div class="issue-card issue-{issue['type'].lower()}">
                        <span class="badge {badge_class}">{issue['type']}</span>
                        <strong style="margin-left: 8px;">{issue['title']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(issue["msg"])

                    # Details Accordion (Grammarly style)
                    with st.expander("🔍 View Impact & Details"):
                        st.caption(f"**Impact:** {issue['impact']}")

                    btn1, btn2 = st.columns(2)
                    with btn1:
                        if st.button("Accept", key=f"acc_{issue['id']}", use_container_width=True):
                            apply_fix_by_id(issue["id"])
                            st.rerun()
                    with btn2:
                        if st.button("Dismiss", key=f"dis_{issue['id']}", use_container_width=True):
                            st.session_state.ignored_issues.add(issue["id"])
                            st.rerun()
                    
                    st.divider()
    else:
        st.write("Load a dataset to trigger automated Grammarly quality suggestions.")