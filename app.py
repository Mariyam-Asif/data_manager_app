import streamlit as st
import pandas as pd
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from collections import deque

st.set_page_config(page_title="⚡ Smart Data Manager", layout='wide')
st.title("⚡ Smart Data Manager")
st.write("Seamlessly transform and refine your datasets with real-time preview, undo/redo, and advanced visualization!")

if "history" not in st.session_state:
    st.session_state.history = {}
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = {}

uploaded_files = st.file_uploader("Upload Your Dataset (CSV, Excel, JSON, XML):", type=["csv", "xlsx", "json", "xml"],
accept_multiple_files=True)

# File size formats
def format_file_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2: 
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024**3):.2f} GB"

# Convert XML to DataFrame
def parse_xml(file):
    try:
        tree = ET.parse(file)
        root = tree.getroot()

        all_records = []
        for record in root.findall(".//record"): 
            data = {elem.tag: elem.text for elem in record}
            all_records.append(data)

        if not all_records:
            st.error("No valid records found in XML. Please check the structure.")
            return pd.DataFrame()

        return pd.DataFrame(all_records)
    
    except ET.ParseError:
        st.error("Error parsing XML file. Ensure it's a valid XML format.")
        return pd.DataFrame()

if uploaded_files:
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        if file_ext == ".csv":
            df = pd.read_csv(file)
        elif file_ext == ".xlsx":
            df = pd.read_excel(file)
        elif file_ext == ".json":
            df = pd.read_json(file)
        elif file_ext == ".xml":
            df = parse_xml(file)
        else:
            st.error(f"Unsupported file type: {file_ext}")
            continue

        # Store `df` persistently in session_state
        file_key = f"df_{file.name}" #Unique key for each file

        # Ensure history and redo stacks exist for this file
        if file_key not in st.session_state.history:
            st.session_state.history[file_key] = deque()
        if file_key not in st.session_state.redo_stack:
            st.session_state.redo_stack[file_key] = deque()
        if file_key not in st.session_state:
            st.session_state[file_key] = df.copy()
        
        # Use the stored Dataframe
        df = st.session_state[file_key]

        # Get actual file size
        file_size = file.getbuffer().nbytes

        # Display name and dynamic file size
        st.write(f"**File Name:** {file.name}")
        st.write(f"**File Size:** {format_file_size(file_size)}")

        # Data Preview
        with st.expander("🔍 Quick Preview of the Data", expanded=False):
            st.dataframe(df.head())

        # Data Cleaning Options
        st.subheader("🛠️ Data Cleaning Options")
        if st.checkbox(f"Enable Cleaning for {file.name}"):
            
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Remove Duplicates"):
                    if file_key in st.session_state:
                        st.session_state.history[file_key].append(st.session_state[file_key].copy())
                        st.session_state[file_key] = st.session_state[file_key].drop_duplicates()
                        st.session_state.redo_stack[file_key].clear()
                        st.toast("✅ Duplicates removed successfully!", icon="✅")
            with col2:
                if st.button("Fill Missing Values"):
                    if file_key in st.session_state:
                        st.session_state.history[file_key].append(st.session_state[file_key].copy())
                        nums_cols = st.session_state[file_key].select_dtypes(include=['number']).columns
                        st.session_state[file_key][nums_cols] = st.session_state[file_key][nums_cols].fillna(
                            st.session_state[file_key][nums_cols].mean()
                        )
                        st.session_state.redo_stack[file_key].clear()
                        st.toast("✅ Missing values filled with column means!", icon="✅")
            with col3:
                if st.button("Undo") and st.session_state.history[file_key]:
                    st.session_state.redo_stack[file_key].append(st.session_state[file_key].copy())
                    st.session_state[file_key] = st.session_state.history[file_key].pop()
                    st.toast("Undo applied!")

                if st.button("Redo") and st.session_state.redo_stack[file_key]:
                    st.session_state.history[file_key].append(st.session_state[file_key].copy())
                    st.session_state[file_key] = st.session_state.redo_stack[file_key].pop()
                    st.toast("Redo applied!")
            
            st.dataframe(st.session_state[file_key].head())

        # Choose Specific Columns to Keep or Convert
        st.subheader("🎯 Customize Columns")
        selected_columns = st.multiselect(f"Choose Columns for {file.name}", df.columns, default=df.columns)
        st.session_state[file_key] = df[selected_columns]

        # Data Visualizations
        st.subheader("📊 Data Insights")
        if st.checkbox(f"Enable Visualization for {file.name}"):
            numeric_cols = df.select_dtypes(include='number')
            categorial_cols = df.select_dtypes(exclude='number')

            #Numeric Data: Line Chart
            if not numeric_cols.empty:
                st.write("📈 **Numeric Data Visualization**")
                st.line_chart(numeric_cols)

            # Categorical Data: Count Plots 
            if not categorial_cols.empty:
                st.write("📊 **Categorical Data Distribution**")  
                for col in categorial_cols.columns[:2]:
                    if df[col].nunique() > 1:
                        fig, ax = plt.subplots()  
                        sns.countplot(y=df[col], ax=ax, palette="coolwarm")
                        st.pyplot(fig)      

        # File Conversion
        st.subheader("🔁 Convert & Download")
        conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel", "JSON", "XML"], key=file.name)
        
        def convert_to_xml(df):
            root = ET.Element("data")
            for _, row in df.iterrows():
                record = ET.SubElement(root, "record")
                for col in df.columns:
                    field = ET.SubElement(record, col)
                    field.text = str(row[col]) if pd.notna(row[col]) else "N/A"
            
            return ET.tostring(root, encoding="utf-8")
            
        def convert_to_json(df):
                return df.to_json(orient="records", indent=4).encode("utf-8")

        if st.button(f"Convert {file.name}"):
            buffer = BytesIO()
            if conversion_type == "CSV":
                df.to_csv(buffer,index=False)
                output_filename = file.name.replace(file_ext, ".csv")
                mime_type = "text/csv"

            elif conversion_type == "Excel":
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False)
                    writer.close()
                output_filename = file.name.replace(file_ext, ".xlsx")
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            elif conversion_type == "JSON":
                buffer.write(convert_to_json(df))
                output_filename = file.name.replace(file_ext, ".json")
                mime_type = "application/json"
            
            elif conversion_type == "XML":
                buffer.write(convert_to_xml(df))
                output_filename = file.name.replace(file_ext, ".xml")
                mime_type = "application/xml"
            
            buffer.seek(0)

            # Store processing state 
            st.success(f"✅ {file.name} has been processed successfully! Click below to download.")

            # Download Button
            st.download_button(
                 label=f"⬇️ Download {file.name} as {conversion_type}",
                data=buffer,
                file_name=output_filename,
                mime=mime_type
            )
