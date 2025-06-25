import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Excel Sheet Editor", layout="wide")
st.title("📊 Chat Excel Sheet Editor")

st.markdown("""
Welcome! Upload an Excel file, view and edit its sheets, and download your changes.\
All processing is done locally for your privacy and safety.
""")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        sheet = st.selectbox("Select a sheet to view/edit", sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.success(f"Editing: {sheet}")

        # Download edited file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name=sheet)
        st.download_button(
            label="Download Edited Excel File",
            data=output.getvalue(),
            file_name="edited_" + uploaded_file.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload an Excel (.xlsx) file to get started.") 