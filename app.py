import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="DataGrammarly - AI Data Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for data persistence
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []

def generate_sample_data():
    """Generates a messy sample dataset to demonstrate the AI checks."""
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

def analyze_dataset(df):
    """Performs automated data health check and computes score."""
    issues = []
    score = 100

    # 1. Missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append({"type": "Critical", "msg": f"{missing_count} missing values detected.", "fix": "impute_missing"})
        score -= 15

    # 2. Duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append({"type": "High", "msg": f"{dupes} duplicate row(s) found.", "fix": "remove_duplicates"})
        score -= 20

    # 3. Negative numeric values where unexpected
    if "Age" in df.columns and (df["Age"] < 0).any():
        issues.append({"type": "Critical", "msg": "Negative values detected in 'Age' column.", "fix": "fix_negative_age"})
        score -= 15

    # 4. Text whitespace issues
    if "Full Name" in df.columns:
        whitespace_count = df["Full Name"].dropna().apply(lambda x: x != x.strip()).sum()
        if whitespace_count > 0:
            issues.append({"type": "Low", "msg": f"{whitespace_count} names have leading/trailing whitespace.", "fix": "trim_names"})
            score -= 10

    return max(0, score), issues

# --- UI Layout ---
st.title("📊 DataGrammarly AI Assistant")
st.markdown("Proactively analyze, clean, and score your structured datasets like Grammarly.")

# Sidebar - Ingestion & Controls
with st.sidebar:
    st.header("📁 Data Source")
    upload_option = st.radio("Choose source:", ["Use Sample Messy Data", "Upload CSV"])
    
    if upload_option == "Use Sample Messy Data":
        if st.button("Load Sample Dataset"):
            st.session_state.df = generate_sample_data()
            st.session_state.history = ["Loaded initial messy dataset."]
            st.success("Sample data loaded!")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.history = ["Loaded uploaded CSV."]
            st.success("File uploaded successfully!")

    st.markdown("---")
    st.header("💡 Grammarly Suggestions")
    
    if st.session_state.df is not None:
        score, issues = analyze_dataset(st.session_state.df)
        st.metric(label="Data Health Score", value=f"{score}/100")
        
        if not issues:
            st.success("✨ Dataset looks pristine! No major issues found.")
        else:
            for idx, issue in enumerate(issues):
                color = "red" if issue["type"] == "Critical" else "orange" if issue["type"] == "High" else "blue"
                st.markdown(f":{color}[**[{issue['type']}]**] {issue['msg']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply Fix", key=f"apply_{idx}"):
                        if issue["fix"] == "remove_duplicates":
                            st.session_state.df = st.session_state.df.drop_duplicates()
                            st.session_state.history.append("Removed duplicate rows.")
                        elif issue["fix"] == "impute_missing":
                            for col in st.session_state.df.columns:
                                if st.session_state.df[col].dtype in [np.float64, np.int64]:
                                    st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].median())
                                else:
                                    st.session_state.df[col] = st.session_state.df[col].fillna("Unknown")
                            st.session_state.history.append("Imputed missing values.")
                        elif issue["fix"] == "fix_negative_age":
                            st.session_state.df["Age"] = st.session_state.df["Age"].apply(lambda x: abs(x) if pd.notnull(x) else x)
                            st.session_state.history.append("Converted negative ages to absolute values.")
                        elif issue["fix"] == "trim_names":
                            st.session_state.df["Full Name"] = st.session_state.df["Full Name"].str.strip()
                            st.session_state.history.append("Trimmed trailing/leading spaces in names.")
                        st.rerun()
                with col2:
                    if st.button("Ignore", key=f"ignore_{idx}"):
                        st.info("Suggestion ignored.")
        
        if st.button("Apply All Fixes"):
            st.session_state.df = st.session_state.df.drop_duplicates()
            st.session_state.df["Age"] = st.session_state.df["Age"].apply(lambda x: abs(x) if pd.notnull(x) else x)
            if "Full Name" in st.session_state.df.columns:
                st.session_state.df["Full Name"] = st.session_state.df["Full Name"].str.strip()
            st.session_state.history.append("Applied all automated cleanups.")
            st.success("All fixes applied successfully!")
            st.rerun()

# Main Panel Display
if st.session_state.df is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Dataset View", "📊 Summary Statistics", "📜 Change Audit History"])
    
    with tab1:
        st.subheader("Current Dataset Preview")
        st.dataframe(st.session_state.df, use_container_width=True)
        st.caption(f"Rows: {st.session_state.df.shape[0]} | Columns: {st.session_state.df.shape[1]}")
        
    with tab2:
        st.subheader("Statistical Profile")
        st.write(st.session_state.df.describe(include="all"))
        
    with tab3:
        st.subheader("Audit Trail & Version Control")
        if not st.session_state.history:
            st.write("No actions recorded yet.")
        else:
            for step in st.session_state.history:
                st.markdown(f"- {step}")
else:
    st.info("👉 Get started by clicking **'Load Sample Dataset'** or uploading a CSV file in the sidebar.")