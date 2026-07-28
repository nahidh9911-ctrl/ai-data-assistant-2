import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="DataGrammarly Assistant", page_icon="✨", layout="wide")

# State management
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []

def find_column(df, keywords):
    """Crash-proof column lookup for any CSV headers."""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in str(col).strip().lower():
                return col
    return None

def analyze_data(df):
    """Generates Grammarly-style quality insights."""
    issues = []
    score = 100

    # Null value check
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        issues.append({
            "id": "nulls", 
            "type": "Critical", 
            "title": "Missing Data", 
            "msg": f"Found **{null_count} missing entries** across the dataset."
        })
        score -= 20

    # Duplicate check
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append({
            "id": "dupes", 
            "type": "High", 
            "title": "Duplicate Rows", 
            "msg": f"Found **{dupes} identical duplicate row(s)**."
        })
        score -= 20

    # Negative age check
    age_col = find_column(df, ["age"])
    if age_col:
        neg_count = (pd.to_numeric(df[age_col], errors='coerce') < 0).sum()
        if neg_count > 0:
            issues.append({
                "id": "negative_age", 
                "type": "Critical", 
                "title": f"Invalid Values in '{age_col}'", 
                "msg": f"Found **{neg_count} negative value(s)** in age."
            })
            score -= 15

    # Whitespace check
    text_cols = df.select_dtypes(include=['object']).columns
    ws_found = False
    for col in text_cols:
        ws_count = df[col].dropna().astype(str).apply(lambda x: x != x.strip()).sum()
        if ws_count > 0:
            ws_found = True
            break
    if ws_found:
        issues.append({
            "id": "spaces", 
            "type": "Low", 
            "title": "Untrimmed Spaces", 
            "msg": "Found extra leading or trailing spaces in text columns."
        })
        score -= 10

    return max(0, score), issues

# App Header
st.title("✨ DataGrammarly Assistant")
st.caption("Professional AI Data Cleaner & Quality Monitor")

# File Upload Section
uploaded = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded is not None and st.session_state.df is None:
    st.session_state.df = pd.read_csv(uploaded)
    st.session_state.history = ["Uploaded new CSV dataset."]
    st.rerun()

# Main App Experience
if st.session_state.df is not None:
    df = st.session_state.df
    score, issues = analyze_data(df)

    # Sidebar Grammarly Panel
    with st.sidebar:
        st.header("💡 Grammarly Insights")
        
        # Health Score Indicator
        if score >= 80:
            st.success(f"🟢 Data Health: {score}/100")
        elif score >= 50:
            st.warning(f"🟡 Data Health: {score}/100")
        else:
            st.error(f"🔴 Data Health: {score}/100")

        st.divider()

        if not issues:
            st.success("🎉 Dataset is clean! No issues detected.")
        else:
            st.write(f"**{len(issues)} Issues Detected:**")
            for issue in issues:
                st.subheader(f"[{issue['type']}] {issue['title']}")
                st.markdown(issue['msg'])
                st.divider()

            if st.button("⚡ Apply All Fixes", type="primary", use_container_width=True):
                # Working on a clean copy of the dataframe
                cleaned_df = st.session_state.df.copy()

                # 1. Deduplicate
                cleaned_df = cleaned_df.drop_duplicates()
                
                # 2. Fix negative ages
                age_col = find_column(cleaned_df, ["age"])
                if age_col:
                    cleaned_df[age_col] = pd.to_numeric(cleaned_df[age_col], errors='coerce').abs()

                # 3. Strip text whitespace
                for col in cleaned_df.select_dtypes(include=['object']).columns:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

                # 4. Fill missing values safely
                for col in cleaned_df.columns:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna("Unknown")

                # Overwrite session state with cleaned data
                st.session_state.df = cleaned_df
                st.session_state.history.append("Fixed all duplicate, missing, and formatting issues.")
                st.rerun()

        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.history = []
            st.rerun()

    # Main Dataset Display
    tab1, tab2 = st.tabs(["📋 Data Preview", "📜 Audit Log"])
    
    with tab1:
        st.dataframe(st.session_state.df, use_container_width=True)
        csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", data=csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

    with tab2:
        for log in st.session_state.history:
            st.write(f"✔️ {log}")

else:
    st.info("👆 Upload your CSV file to get started.")