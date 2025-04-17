import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import re
import nltk
from textblob import TextBlob
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import docx2txt
import PyPDF2
import concurrent.futures
import os
from PIL import Image
import numpy as np
import base64
import tempfile


# Set up Gemini API
genai.configure(api_key="AIzaSyBj1BzzNCg6FOUeic8DTtU3uYNVMaDErQw")

# Streamlit UI Setup
st.set_page_config(page_title="PRD Analyzer", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .dataframe {
        font-size: 0.8rem;
    }
    .score-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📜 PRD Analyzer - AI Validation</h1>', unsafe_allow_html=True)

# Initialize session state for caching
if 'prd_reviews' not in st.session_state:
    st.session_state.prd_reviews = []
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

# File uploader with multiple formats
uploaded_file = st.file_uploader("Upload your PRD file", type=["xlsx", "docx", "pdf", "txt"])

# Select AI Persona with descriptions
persona_descriptions = {
    "Product Manager": "Focuses on market opportunity, user needs, and business value",
    "UX Designer": "Evaluates user experience, flows, and design consistency",
    "Engineer": "Assesses technical feasibility, architecture, and implementation details",
    "General Analyst": "Provides comprehensive analysis across all dimensions"
}
persona = st.selectbox(
    "Choose AI Persona:", 
    list(persona_descriptions.keys()),
    format_func=lambda x: f"{x} - {persona_descriptions[x]}"
)

# Function to extract text from different file formats
def extract_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.xlsx':
        try:
            df = pd.read_excel(file)
            return df, "excel"
        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
            return None, None
            
    elif file_extension == '.docx':
        try:
            text = docx2txt.process(file)
            # Convert to DataFrame with a single entry
            df = pd.DataFrame({"Content": [text], "Title": [file.name]})
            return df, "docx"
        except Exception as e:
            st.error(f"Error processing Word file: {e}")
            return None, None
            
    elif file_extension == '.pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            # Convert to DataFrame with a single entry
            df = pd.DataFrame({"Content": [text], "Title": [file.name]})
            return df, "pdf"
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return None, None
            
    elif file_extension == '.txt':
        try:
            text = file.read().decode('utf-8')
            # Convert to DataFrame with a single entry
            df = pd.DataFrame({"Content": [text], "Title": [file.name]})
            return df, "txt"
        except Exception as e:
            st.error(f"Error processing text file: {e}")
            return None, None
    
    return None, None

# Function to split long text into manageable chunks
def chunk_text(text, max_length=8000):
    """Split text into chunks of max_length characters."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:  # +1 for space
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to analyze a single PRD with Gemini
def analyze_prd(prd_details, prd_title, persona):
    prompt = f"""
    You are an AI acting as a {persona}. Analyze this PRD and provide:
    1. **Score** (0-100%)
    2. **SWOT Analysis**
    3. **Key Missing Elements & Suggestions**
    4. **Rewrite missing sections to improve clarity**
    5. **Convert suggestions into actionable tasks**
    PRD Title: {prd_title}
    PRD Details:
    {prd_details}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Extract score
    match = re.search(r"(\d{1,3})\s*[/|%]?\s*100?", response.text)
    score = int(match.group(1)) if match else 50

    # Calculate sentiment
    sentiment = TextBlob(prd_details).sentiment.polarity
    
    return {
        "title": prd_title,
        "score": score,
        "sentiment": sentiment,
        "feedback": response.text
    }

# Analyze PRD data in parallel
def analyze_prd_data(df, file_type):
    prd_reviews = []
    
    # Setup processing status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each row based on file type
    if file_type == "excel":
        # For Excel files, we analyze each row
        total_rows = len(df)
        
        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_prd = {}
            
            for index, row in df.iterrows():
                prd_title = row["Title"] if "Title" in df.columns and pd.notna(row["Title"]) else f"PRD {index+1}"
                
                # Prepare text content based on available columns
                if "Content" in df.columns and pd.notna(row["Content"]):
                    prd_text = row["Content"]
                else:
                    prd_text = row.to_string()
                
                # Submit task to thread pool
                future = executor.submit(analyze_prd, prd_text, prd_title, persona)
                future_to_prd[future] = index
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_prd)):
                result = future.result()
                prd_reviews.append(result)
                
                # Update progress
                progress = (i + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{total_rows} PRDs...")
    
    else:
        # For other file types, we analyze the entire document
        status_text.text("Processing document...")
        
        for index, row in df.iterrows():
            prd_title = row["Title"] if "Title" in df.columns else f"Document {index+1}"
            prd_text = row["Content"] if "Content" in df.columns else row.to_string()
            
            # Chunk the text if it's too long
            chunks = chunk_text(prd_text)
            
            if len(chunks) > 1:
                # Process each chunk separately and combine results
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Processing document chunk {i+1}/{len(chunks)}...")
                    progress_bar.progress((i+1)/len(chunks))
                    result = analyze_prd(chunk, f"{prd_title} - Part {i+1}", persona)
                    chunk_results.append(result)
                
                # Combine results from all chunks
                combined_score = sum(r["score"] for r in chunk_results) / len(chunk_results)
                combined_sentiment = sum(r["sentiment"] for r in chunk_results) / len(chunk_results)
                combined_feedback = "\n\n".join([f"PART {i+1}:\n{r['feedback']}" for i, r in enumerate(chunk_results)])
                
                prd_reviews.append({
                    "title": prd_title,
                    "score": combined_score,
                    "sentiment": combined_sentiment,
                    "feedback": combined_feedback
                })
            else:
                # Process single chunk
                progress_bar.progress(0.5)
                result = analyze_prd(prd_text, prd_title, persona)
                prd_reviews.append(result)
                progress_bar.progress(1.0)
    
    status_text.text("Analysis completed!")
    return prd_reviews

# Main app logic
if uploaded_file:
    if not st.session_state.file_processed:
        with st.spinner("Processing your file..."):
            df, file_type = extract_text_from_file(uploaded_file)
            if df is not None:
                st.session_state.parsed_data = df
                st.session_state.file_type = file_type
                st.session_state.file_processed = True
    
    if st.session_state.parsed_data is not None:
        df = st.session_state.parsed_data
        file_type = st.session_state.file_type
        
        st.markdown('<h2 class="section-header">📄 Extracted PRD Data:</h2>', unsafe_allow_html=True)
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Start Analysis", key="start_analysis"):
                with st.spinner("Analyzing your PRD..."):
                    st.session_state.prd_reviews = analyze_prd_data(df, file_type)
        
        with col2:
            if st.button("🔄 Reset Analysis", key="reset_analysis"):
                st.session_state.prd_reviews = []
                st.session_state.file_processed = False
                st.experimental_rerun()
        
        # Display analysis results
        if st.session_state.prd_reviews:
            st.markdown('<h2 class="section-header">🏆 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Score Cards", "Detailed Analysis", "Visualizations", "Export"])
            
            with tab1:
                for review in st.session_state.prd_reviews:
                    with st.container():
                        st.markdown(f"### 🏆 {review['title']}")
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"<div class='score-container'>", unsafe_allow_html=True)
                            score = int(review['score'])
                            score_bar = st.progress(0)
                            for i in range(score):
                                time.sleep(0.005)  # Faster animation
                                score_bar.progress(i + 1)
                            st.write(f"### 🎯 Final Score: {score}%")
                            
                            sentiment = review['sentiment']
                            sentiment_desc = "Positive" if sentiment > 0.1 else "Neutral" if sentiment > -0.1 else "Negative"
                            st.write(f"📊 **Sentiment Analysis:** {sentiment_desc} ({sentiment:.2f})")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            if score >= 80:
                                st.success("Excellent PRD!")
                            elif score >= 60:
                                st.info("Good PRD with room for improvement")
                            else:
                                st.warning("PRD needs significant work")
            
            with tab2:
                for review in st.session_state.prd_reviews:
                    with st.expander(f"📝 {review['title']} - Feedback"):
                        st.markdown(review['feedback'])
            
            with tab3:
                # Score comparison chart
                scores = [r['score'] for r in st.session_state.prd_reviews]
                titles = [r['title'] for r in st.session_state.prd_reviews]
                
                # Truncate long titles
                short_titles = [t[:20] + "..." if len(t) > 20 else t for t in titles]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(short_titles, scores, color=plt.cm.viridis(np.linspace(0, 1, len(scores))))
                ax.set_xlabel("Score (%)")
                ax.set_title("PRD Score Comparison")
                ax.set_xlim(0, 100)
                
                # Add score values at the end of each bar
                for i, v in enumerate(scores):
                    ax.text(v + 2, i, f"{v:.1f}%", va='center')
                
                st.pyplot(fig)
                
                # Sentiment analysis chart
                sentiments = [r['sentiment'] for r in st.session_state.prd_reviews]
                
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                bars = ax2.barh(short_titles, sentiments, 
                               color=[plt.cm.RdYlGn((s+1)/2) for s in sentiments])
                ax2.set_xlabel("Sentiment (-1 to 1)")
                ax2.set_title("PRD Sentiment Analysis")
                ax2.set_xlim(-1, 1)
                
                # Add sentiment values at the end of each bar
                for i, v in enumerate(sentiments):
                    ax2.text(v + 0.05 if v >= 0 else v - 0.15, i, f"{v:.2f}", va='center')
                
                st.pyplot(fig2)
            
            with tab4:
                # Export Analysis to Excel
                def export_excel():
                    output = BytesIO()
                    export_df = pd.DataFrame([
                        {
                            "Title": r["title"],
                            "Score": r["score"],
                            "Sentiment": r["sentiment"],
                            "AI Feedback": r["feedback"]
                        } for r in st.session_state.prd_reviews
                    ])
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False, sheet_name="PRD_Analysis")
                        
                        # Add chart sheet
                        workbook = writer.book
                        chart_sheet = workbook.add_worksheet("Score_Chart")
                        
                        # Add scores chart
                        chart = workbook.add_chart({'type': 'bar'})
                        chart.add_series({
                            'name': 'PRD Scores',
                            'categories': f'=PRD_Analysis!$A$2:$A${len(st.session_state.prd_reviews)+1}',
                            'values': f'=PRD_Analysis!$B$2:$B${len(st.session_state.prd_reviews)+1}',
                        })
                        chart.set_title({'name': 'PRD Score Comparison'})
                        chart.set_x_axis({'name': 'Score (%)'})
                        chart_sheet.insert_chart('A1', chart)
                    
                    return output.getvalue()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📥 Download Excel Report", 
                        export_excel(), 
                        file_name="PRD_Analysis.xlsx",
                        help="Download detailed analysis in Excel format"
                    )
                    
                with col2:
                    # Export Analysis to PDF
                    def export_pdf():
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", style='B', size=16)
                        pdf.cell(200, 10, "PRD Analysis Report", ln=True, align='C')
                        pdf.ln(10)
                        
                        for review in st.session_state.prd_reviews:
                            pdf.set_font("Arial", style='B', size=12)
                            pdf.cell(0, 10, review["title"], ln=True)
                            pdf.set_font("Arial", size=10)
                            pdf.multi_cell(0, 10, f"Score: {review['score']}%\nSentiment: {review['sentiment']:.2f}\n")
                            pdf.ln(5)
                            pdf.multi_cell(0, 10, f"Feedback Summary:\n{review['feedback'][:800]}...")
                            pdf.ln(10)
                        
                        # Add visualization
                        if len(st.session_state.prd_reviews) > 0:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            y_pos = np.arange(len(titles))
                            ax.barh(y_pos, scores, align='center')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(short_titles)
                            ax.set_xlabel('Score (%)')
                            ax.set_title('PRD Scores')
                            plt.tight_layout()
                            
                            # Save chart to temp file
                            chart_file = BytesIO()
                            plt.savefig(chart_file, format='png')
                            chart_file.seek(0)
                            
                            # Add chart to PDF
                            # Add chart to PDF
                            pdf.add_page()
                            pdf.set_font("Arial", style='B', size=14)
                            pdf.cell(200, 10, "PRD Score Comparison", ln=True, align='C')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                                tmp_img.write(chart_file.getvalue())
                                tmp_img_path = tmp_img.name
                                pdf.image(tmp_img_path, x=20, y=30, w=170)
                                os.remove(tmp_img_path)  # Clean up
                                st.download_button(
                                    export_pdf(), 
                                    file_name="PRD_Analysis.pdf",
                                    help="Download detailed analysis in PDF format"
                                    )
                
                # Generate AI-enhanced PRD
                st.markdown("### 🌟 Generate Enhanced PRD")
                selected_review = st.selectbox(
                    "Select a PRD to enhance:", 
                    [r["title"] for r in st.session_state.prd_reviews]
                )
                
                if selected_review and st.button("Generate Enhanced PRD"):
                    # Find the selected review
                    selected_data = next((r for r in st.session_state.prd_reviews if r["title"] == selected_review), None)
                    
                    if selected_data:
                        with st.spinner("Generating enhanced PRD..."):
                            # Generate enhanced PRD
                            model = genai.GenerativeModel("gemini-1.5-flash")
                            prompt = f"""
                            Based on the analysis and feedback:
                            {selected_data['feedback']}
                            
                            Generate a complete, enhanced PRD that addresses all issues and incorporates 
                            all suggestions. Format it using proper markdown with clear sections and subsections.
                            """
                            
                            response = model.generate_content(prompt)
                            
                            # Display the enhanced PRD
                            st.markdown("## Enhanced PRD")
                            st.markdown(response.text)
                            
                            # Add download button for enhanced PRD
                            st.download_button(
                                "📥 Download Enhanced PRD", 
                                response.text, 
                                file_name=f"Enhanced_{selected_review}.md",
                                mime="text/markdown",
                                help="Download the enhanced PRD as a markdown file"
                            )

# Add a sidebar with additional information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/document.png")
    st.markdown("## About PRD Analyzer")
    st.markdown("""
    This tool analyzes Product Requirement Documents (PRDs) using AI to provide:
    
    - Score assessment
    - SWOT analysis
    - Missing elements identification
    - Actionable improvement tasks
    - Enhanced PRD generation
    
    **Supported file formats:**
    - Excel (.xlsx)
    - Word (.docx)
    - PDF (.pdf)
    - Text (.txt)
    """)
    

    
    st.markdown("---")
    

# Create a custom layout for the main app
if "view" not in st.session_state:
    st.session_state.view = "analyzer"

# Add navigation in the sidebar

# Function to create PRD templates
def generate_prd_template(template_type):
    templates = {
        "Standard PRD": """# Product Requirements Document

## 1. Introduction
### 1.1 Purpose
### 1.2 Scope
### 1.3 Definitions and Acronyms

## 2. Product Overview
### 2.1 Product Perspective
### 2.2 Product Features
### 2.3 User Classes and Characteristics

## 3. Specific Requirements
### 3.1 Functional Requirements
### 3.2 Non-functional Requirements
### 3.3 Technical Requirements

## 4. User Interface
### 4.1 User Interfaces
### 4.2 Hardware Interfaces
### 4.3 Software Interfaces

## 5. System Features
### 5.1 Feature 1
### 5.2 Feature 2

## 6. Other Non-functional Requirements
### 6.1 Performance Requirements
### 6.2 Safety Requirements
### 6.3 Security Requirements

## 7. Appendix
""",
        "Agile PRD": """# Agile Product Requirements

## 1. Vision
### 1.1 Problem Statement
### 1.2 Solution Approach
### 1.3 Key Metrics

## 2. Personas
### 2.1 Primary Persona
### 2.2 Secondary Persona

## 3. User Stories
### 3.1 Epic 1
- User Story 1.1
- User Story 1.2
### 3.2 Epic 2
- User Story 2.1
- User Story 2.2

## 4. Feature List
### 4.1 MVP Features
### 4.2 Future Enhancements

## 5. Prioritization
### 5.1 MoSCoW Analysis
- Must Have
- Should Have
- Could Have
- Won't Have

## 6. Acceptance Criteria
### 6.1 Feature 1 Acceptance Criteria
### 6.2 Feature 2 Acceptance Criteria

## 7. Technical Considerations
""",
        "Lean PRD": """# Lean Product Requirements

## 1. Problem
### 1.1 Customer Pain Points
### 1.2 Current Alternatives

## 2. Solution
### 2.1 Value Proposition
### 2.2 Minimum Viable Product

## 3. Key Features
### 3.1 Feature 1
### 3.2 Feature 2

## 4. Success Metrics
### 4.1 KPIs
### 4.2 Measurement Plan

## 5. Timeline
### 5.1 Release Plan
### 5.2 Milestones
"""
    }
    
    return templates.get(template_type, "")

# Function for historical analysis tracking
def save_analysis_history(analysis_data):
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": timestamp,
        "data": analysis_data,
        "username": st.session_state.get("username", "Anonymous")
    })

# Function for batch processing
def batch_process_files(files):
    results = []
    
    for file in files:
        df, file_type = extract_text_from_file(file)
        if df is not None:
            # Process each file
            file_results = analyze_prd_data(df, file_type)
            results.append({
                "filename": file.name,
                "results": file_results
            })
    
    return results

# Function to compare two PRDs and generate recommendations
def compare_prds(prd1, prd2):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Compare these two PRDs and provide:
    1. Strengths of each PRD
    2. Common elements
    3. Unique elements in each
    4. Recommendations for improvements
    
    PRD 1: {prd1}
    
    PRD 2: {prd2}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Function to generate competitive analysis
def generate_competitive_analysis(prd_text, market_data=None):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    market_context = ""
    if market_data:
        market_context = f"Consider this market data: {market_data}"
    
    prompt = f"""
    Based on this PRD, generate a competitive analysis including:
    1. Key competitors in this space
    2. Feature comparison matrix
    3. Market positioning
    4. Competitive advantages and disadvantages
    5. Strategy recommendations
    
    PRD: {prd_text}
    
    {market_context}
    """
    
    response = model.generate_content(prompt)
    return response.text

# Main app logic based on navigation selection
if st.session_state.view == "analyzer":
    # Main PRD Analyzer view - existing functionality remains here
    pass

elif st.session_state.view == "batch":
    st.markdown('<h1 class="main-header">🔄 Batch PRD Processing</h1>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload multiple PRD files", type=["xlsx", "docx", "pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        if st.button("Process All Files"):
            with st.spinner("Processing batch of files..."):
                batch_results = batch_process_files(uploaded_files)
                
                # Store results in session state
                st.session_state.batch_results = batch_results
                
                # Show summary
                st.success(f"Successfully processed {len(batch_results)} files")
                
                # Display summary table
                summary_data = []
                for file_result in batch_results:
                    filename = file_result["filename"]
                    avg_score = sum(r["score"] for r in file_result["results"]) / len(file_result["results"]) if file_result["results"] else 0
                    summary_data.append({
                        "Filename": filename,
                        "Average Score": f"{avg_score:.1f}%",
                        "PRDs Analyzed": len(file_result["results"])
                    })
                
                st.markdown('<h2 class="section-header">📊 Batch Processing Summary</h2>', unsafe_allow_html=True)
                st.table(pd.DataFrame(summary_data))
                
                # Option to download consolidated results
                if st.button("Generate Consolidated Report"):
                    # Create consolidated Excel report
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Summary sheet
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                        
                        # Individual file sheets
                        for i, file_result in enumerate(batch_results):
                            filename = file_result["filename"]
                            sheet_name = f"File_{i+1}"
                            
                            # Create dataframe for this file
                            file_df = pd.DataFrame([
                                {
                                    "Title": r["title"],
                                    "Score": r["score"],
                                    "Sentiment": r["sentiment"]
                                } for r in file_result["results"]
                            ])
                            
                            file_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    st.download_button(
                        "📥 Download Consolidated Report",
                        output.getvalue(),
                        file_name="Batch_PRD_Analysis.xlsx"
                    )
        
        # Display individual file results if available
        if 'batch_results' in st.session_state:
            for file_result in st.session_state.batch_results:
                with st.expander(f"📁 {file_result['filename']} Results"):
                    for prd_result in file_result["results"]:
                        st.markdown(f"### {prd_result['title']}")
                        st.write(f"Score: {prd_result['score']}%")
                        st.write(f"Sentiment: {prd_result['sentiment']:.2f}")
                        
                        with st.expander("Show Full Analysis"):
                            st.markdown(prd_result["feedback"])

elif st.session_state.view == "templates":
    st.markdown('<h1 class="main-header">📝 PRD Templates</h1>', unsafe_allow_html=True)
    
    template_type = st.selectbox(
        "Select a template type:",
        ["Standard PRD", "Agile PRD", "Lean PRD"]
    )
    
    template_content = generate_prd_template(template_type)
    
    st.markdown("### Template Preview")
    st.markdown(template_content)
    
    # Download template
    st.download_button(
        "📥 Download Template",
        template_content,
        file_name=f"{template_type.replace(' ', '_')}.md",
        mime="text/markdown"
    )
    
    # Custom template builder
    st.markdown("### Custom Template Builder")
    st.markdown("Select sections to include in your custom template:")
    
    # Template sections
    sections = {
        "Introduction": st.checkbox("Introduction", value=True),
        "Product Overview": st.checkbox("Product Overview", value=True),
        "User Personas": st.checkbox("User Personas"),
        "User Stories": st.checkbox("User Stories"),
        "Features": st.checkbox("Features", value=True),
        "Non-functional Requirements": st.checkbox("Non-functional Requirements"),
        "UI/UX": st.checkbox("UI/UX"),
        "Technical Architecture": st.checkbox("Technical Architecture"),
        "Data Requirements": st.checkbox("Data Requirements"),
        "Success Metrics": st.checkbox("Success Metrics"),
        "Timeline": st.checkbox("Timeline"),
        "Risks and Mitigations": st.checkbox("Risks and Mitigations"),
        "Appendix": st.checkbox("Appendix")
    }
    
    if st.button("Generate Custom Template"):
        custom_template = "# Custom Product Requirements Document\n\n"
        
        for section, include in sections.items():
            if include:
                custom_template += f"## {section}\n"
                custom_template += "### [Add content here]\n\n"
        
        st.markdown("### Your Custom Template")
        st.text_area("Custom Template", custom_template, height=400)
        
        st.download_button(
            "📥 Download Custom Template",
            custom_template,
            file_name="Custom_PRD_Template.md",
            mime="text/markdown"
        )

elif st.session_state.view == "history":
    st.markdown('<h1 class="main-header">📜 Analysis History</h1>', unsafe_allow_html=True)
    
    # Check if history exists
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("No analysis history available yet. Run some analyses first.")
    else:
        # Display history entries
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"Analysis {i+1} - {entry['timestamp']} by {entry['username']}"):
                st.write("Results:")
                for result in entry['data']:
                    st.markdown(f"#### {result['title']}")
                    st.write(f"Score: {result['score']}%")
                    st.write(f"Sentiment: {result['sentiment']:.2f}")
                    
                    with st.expander("Show Feedback"):
                        st.markdown(result["feedback"])
        
        # Option to clear history
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared")
            st.experimental_rerun()

# Add additional modules to the analyzer view
if st.session_state.view == "analyzer" and 'prd_reviews' in st.session_state and st.session_state.prd_reviews:
    st.markdown("---")
    st.markdown('<h2 class="section-header">🧩 Additional Analysis Tools</h2>', unsafe_allow_html=True)
    
    # Create tabs for different additional tools
    tool_tab1, tool_tab2, tool_tab3 = st.tabs(["PRD Comparison", "Competitive Analysis", "Implementation Planning"])
    
    with tool_tab1:
        st.markdown("### Compare Two PRDs")
        st.write("Select two PRDs to compare:")
        
        col1, col2 = st.columns(2)
        with col1:
            prd1_title = st.selectbox("Select first PRD:", [r["title"] for r in st.session_state.prd_reviews], key="prd1_select")
        with col2:
            prd2_title = st.selectbox("Select second PRD:", [r["title"] for r in st.session_state.prd_reviews], key="prd2_select")
        
        if prd1_title != prd2_title and st.button("Compare PRDs"):
            with st.spinner("Generating comparison..."):
                # Get PRD texts
                prd1_data = next((r for r in st.session_state.prd_reviews if r["title"] == prd1_title), None)
                prd2_data = next((r for r in st.session_state.prd_reviews if r["title"] == prd2_title), None)
                
                if prd1_data and prd2_data:
                    comparison_result = compare_prds(prd1_data["feedback"], prd2_data["feedback"])
                    
                    st.markdown("### Comparison Results")
                    st.markdown(comparison_result)
                    
                    # Add download button for comparison
                    st.download_button(
                        "📥 Download Comparison",
                        comparison_result,
                        file_name=f"Comparison_{prd1_title}_vs_{prd2_title}.md",
                        mime="text/markdown"
                    )
    
    with tool_tab2:
        st.markdown("### Competitive Analysis")
        st.write("Generate competitive analysis for a selected PRD:")
        
        selected_prd = st.selectbox("Select PRD:", [r["title"] for r in st.session_state.prd_reviews], key="comp_analysis_select")
        
        # Optional market data input
        market_data = st.text_area("Add market data (optional):", height=100, 
                                   help="Add any relevant market data, competitor information, or industry trends")
        
        if st.button("Generate Competitive Analysis"):
            with st.spinner("Generating competitive analysis..."):
                # Get PRD text
                prd_data = next((r for r in st.session_state.prd_reviews if r["title"] == selected_prd), None)
                
                if prd_data:
                    comp_analysis = generate_competitive_analysis(prd_data["feedback"], market_data)
                    
                    st.markdown("### Competitive Analysis Results")
                    st.markdown(comp_analysis)
                    
                    # Add download button for analysis
                    st.download_button(
                        "📥 Download Competitive Analysis",
                        comp_analysis,
                        file_name=f"Competitive_Analysis_{selected_prd}.md",
                        mime="text/markdown"
                    )
    
    with tool_tab3:
        st.markdown("### Implementation Planning")
        st.write("Generate implementation plan for a selected PRD:")
        
        selected_prd = st.selectbox("Select PRD:", [r["title"] for r in st.session_state.prd_reviews], key="impl_plan_select")
        
        col1, col2 = st.columns(2)
        with col1:
            team_size = st.number_input("Team size:", min_value=1, max_value=100, value=5)
        with col2:
            timeline_months = st.number_input("Timeline (months):", min_value=1, max_value=36, value=6)
        
        if st.button("Generate Implementation Plan"):
            with st.spinner("Generating implementation plan..."):
                # Get PRD text
                prd_data = next((r for r in st.session_state.prd_reviews if r["title"] == selected_prd), None)
                
                if prd_data:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = f"""
                    Based on this PRD analysis, generate a detailed implementation plan including:
                    1. Project phases and timeline (for {timeline_months} months)
                    2. Resource allocation (for a team of {team_size} people)
                    3. Technical dependencies
                    4. Risk assessment
                    5. Success metrics
                    6. Gantt chart (in text format)
                    
                    PRD Analysis: {prd_data["feedback"]}
                    """
                    
                    response = model.generate_content(prompt)
                    implementation_plan = response.text
                    
                    st.markdown("### Implementation Plan")
                    st.markdown(implementation_plan)
                    
                    # Add download button for implementation plan
                    st.download_button(
                        "📥 Download Implementation Plan",
                        implementation_plan,
                        file_name=f"Implementation_Plan_{selected_prd}.md",
                        mime="text/markdown"
                    )
                    
                    # Generate visual Gantt chart if possible
                    st.markdown("### Visual Gantt Chart")
                    try:
                        # Simple Gantt chart using matplotlib
                        tasks = []
                        start_dates = []
                        durations = []
                        
                        # Parse tasks from implementation plan (simplified approach)
                        lines = implementation_plan.split('\n')
                        for line in lines:
                            if "phase" in line.lower() or "milestone" in line.lower():
                                parts = line.split('-')
                                if len(parts) >= 2:
                                    tasks.append(parts[0].strip())
                                    durations.append(int(timeline_months / 4))  # Simplified duration
                                    start_month = len(start_dates) * (timeline_months / 4)
                                    start_dates.append(start_month)
                        
                        # Create Gantt chart
                        if tasks:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Plot tasks
                            y_pos = range(len(tasks))
                            ax.barh(y_pos, durations, left=start_dates, height=0.4)
                            
                            # Customize chart
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(tasks)
                            ax.set_xlabel('Months')
                            ax.set_title('Project Timeline')
                            ax.grid(axis='x')
                            
                            # Add month markers
                            for i in range(timeline_months + 1):
                                ax.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
                            
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not generate visual Gantt chart: {e}")
                        st.info("Please refer to the text-based timeline in the implementation plan.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>PRD Analyzer v2.0 | AI-Powered Document Analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)
