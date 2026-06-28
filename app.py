import streamlit as st
import pandas as pd
import random
import os
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
import base64
from io import BytesIO

st.set_page_config(page_title="NLP-Based Subjective Answer Evaluation System", layout="wide")

# Load BERT model - runs once and caches
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()

# Create dataset directory if it does not exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Initialize session state variables
if 'questions_selected' not in st.session_state:
    st.session_state.questions_selected = False
if 'exam_started' not in st.session_state:
    st.session_state.exam_started = False
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = pd.DataFrame()
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'student_name' not in st.session_state:
    st.session_state.student_name = ""
if 'student_regno' not in st.session_state:
    st.session_state.student_regno = ""
if 'exam_submitted' not in st.session_state:
    st.session_state.exam_submitted = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

@st.cache_data
def load_questions_from_file():
    # Load questions from CSV file
    file_path = "dataset/questions.csv"
    if os.path.exists(file_path):
        try:
            dataframe = pd.read_csv(file_path)
            dataframe.columns = [col.strip() for col in dataframe.columns]
            question_col = None
            answer_col = None

            for col in dataframe.columns:
                if 'question' in col.lower():
                    question_col = col
                if 'answer' in col.lower():
                    answer_col = col

            if question_col and answer_col:
                return dataframe[[question_col, answer_col]].rename(columns={question_col: 'Question', answer_col: 'Answer'})
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def evaluate_with_bert(student_ans, model_ans):
    # Semantic similarity using Sentence BERT
    if not student_ans.strip():
        return 0, 0, "No answer provided"

    embeddings = model.encode([student_ans, model_ans])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarity_percentage = round(similarity * 100, 2)

    # Convert similarity to marks out of 5
    if similarity >= 0.8:
        marks = 5
    elif similarity >= 0.6:
        marks = 4
    elif similarity >= 0.4:
        marks = 3
    elif similarity >= 0.2:
        marks = 2
    elif similarity > 0:
        marks = 1
    else:
        marks = 0

    feedback = f"Semantic similarity: {similarity_percentage}%"
    return marks, similarity_percentage, feedback

def create_pdf(results_df, student_name, regno, total_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)  # Left, Top, Right margin
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Answer Evaluation Report", ln=True, align='C')
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Name: {student_name}  |  RegNo: {regno}", ln=True)
    pdf.cell(0, 8, f"Total Score: {total_score}/50", ln=True)
    pdf.ln(5)
    
    # Body - Epw = Effective Page Width
    epw = pdf.w - 2*pdf.l_margin
    
    for idx, row in results_df.iterrows():
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(epw, 7, f"Q{idx+1}: {str(row['Question'])[:250]}")
        
        pdf.set_font("Arial", '', 10)
        stu_ans = str(row['Your Answer'])[:500].replace('\n', ' ')
        mod_ans = str(row['Model Answer'])[:500].replace('\n', ' ')
        
        pdf.multi_cell(epw, 6, f"Your Answer: {stu_ans}")
        pdf.multi_cell(epw, 6, f"Model Answer: {mod_ans}")
        pdf.multi_cell(epw, 6, f"Marks: {row['Marks']} | Similarity: {row['Similarity']}")
        pdf.ln(4)
    
    return bytes(pdf.output())
# Sidebar for user selection
st.sidebar.title("Select User")
user_type = st.sidebar.selectbox("Choose Role", ["Home", "Teacher", "Student"])

# Main app routing based on user selection
if user_type == "Home":
    st.title("🎓 NLP-Based Subjective Answer Evaluation System")
    st.markdown("### Powered by Sentence BERT")
    st.info("AI Based Subjective Answer Evaluation System with Semantic Understanding")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Feature 1", "Semantic Scoring", "BERT Model")
    with col2:
        st.metric("Feature 2", "Auto Grading", "Real-time")
    with col3:
        st.metric("Feature 3", "PDF Report", "Downloadable")

elif user_type == "Teacher":
    st.title("🧑‍🏫 Teacher Dashboard")
    st.header("1. Upload Questions Dataset")
    uploaded_file = st.file_uploader("Upload Kaggle CSV", type=['csv'], help="CSV must have Question and Answer")

    if uploaded_file is not None:
        with open("dataset/questions.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded Successfully!")
        load_questions_from_file.clear()
        st.rerun()

    questions_df = load_questions_from_file()
    
    if not questions_df.empty:
        st.success(f"✅ Dataset loaded: {len(questions_df)} questions available")
        col1, col2, col3 = st.columns(3)

        with col1:  
            st.metric("📊 Total Questions", len(questions_df))
        
        with col2:
            st.metric("📝 Questions per Exam", 10)
        
        with col3:
            st.metric("🤖 AI Model", "MiniLM")
        
        st.subheader("Preview of Questions")
        st.dataframe(questions_df.head(10), use_container_width=True, height=300)
        
    else:
        st.warning("⚠️ No dataset found. Please upload a CSV file.")

elif user_type == "Student":
    st.title("👨‍🎓 Student Panel")
    questions_df = load_questions_from_file()
    if questions_df.empty:
        st.error("❌ No exam available. Please ask teacher to upload questions first.")
        st.stop()

    # Student details input
    if not st.session_state.exam_started and not st.session_state.exam_submitted:
        st.header("📝 Enter Your Details")
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name", value=st.session_state.student_name)
        with col2:
            student_regno = st.text_input("Register Number", value=st.session_state.student_regno)

        st.session_state.student_name = student_name
        st.session_state.student_regno = student_regno

        st.divider()
        st.header("🚀 Ready to Start Your Exam")
        st.info("You will get 10 random questions. Each question carries 5 marks. Total: 50 marks.")
        st.metric("Total Questions in Database", len(questions_df))

        if st.button("Start Exam", type="primary", use_container_width=True):
            if not student_name or not student_regno:
                st.error("Please enter Name and Register Number")
            elif len(questions_df) >= 10:
                st.session_state.selected_questions = questions_df.sample(n=10).reset_index(drop=True)
                st.session_state.questions_selected = True
                st.session_state.exam_started = True
                st.session_state.student_answers = {}
                st.rerun()
            else:
                st.error("Not enough questions in database. Need at least 10 questions.")

    # Exam interface
    if st.session_state.exam_started and not st.session_state.exam_submitted:
        st.header(f"📋 Exam for {st.session_state.student_name} - {st.session_state.student_regno}")
        st.progress(0.0)
    
    st.warning("⚠️ Do not refresh the page. Answer all questions and submit.")

    with st.form("exam_form"):
        for i, row in st.session_state.selected_questions.iterrows():
            st.subheader(f"Question {i+1}: {row['Question']} [5 Marks]")
            answer = st.text_area(f"Your Answer for Q{i+1}", key=f"student_q{i}", height=100, label_visibility="collapsed")
            st.session_state.student_answers[i] = answer
            st.divider()

        submitted = st.form_submit_button("Submit Exam", type="primary", use_container_width=True)

        if submitted:
            st.session_state.exam_submitted = True
            st.rerun()

    # Results display with BERT scoring
    if st.session_state.exam_submitted:
        st.header("📊 Exam Results")
        st.subheader(f"Student: {st.session_state.student_name} | Reg No: {st.session_state.student_regno}")

        results_data = []
        total_score = 0

        with st.spinner("Evaluating answers using BERT..."):
            for i, row in st.session_state.selected_questions.iterrows():
                student_ans = st.session_state.student_answers.get(i, "")
                model_ans = row['Answer']

                marks, similarity, feedback = evaluate_with_bert(student_ans, model_ans)
                total_score += marks

                results_data.append({
                    "Question": row['Question'],
                    "Your Answer": student_ans,
                    "Model Answer": model_ans,
                    "Marks": f"{marks}/5",
                    "Similarity": f"{similarity}%",
                    "Feedback": feedback
                })

        results_df = pd.DataFrame(results_data)
        st.session_state.results_df = results_df

        # Color code the dataframe
        def color_marks(val):
            mark = int(val.split('/')[0])
            if mark >= 4:
                return 'background-color: #90EE90' # Light green
            elif mark >= 2:
                return 'background-color: #FFD700' # Gold
            else:
                return 'background-color: #FFB6C1' # Light red

        st.dataframe(
            results_df.style.map(color_marks, subset=['Marks']),
            use_container_width=True,
            height=400
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{total_score}/50")
        with col2:
            percentage = round(total_score/50*100, 1)
            st.metric("Percentage", f"{percentage}%")
        with col3:
            if total_score >= 40:
                st.metric("Grade", "A", "Excellent")
            elif total_score >= 25:
                st.metric("Grade", "B", "Good")
            else:
                st.metric("Grade", "C", "Need Practice")

        if total_score >= 40:
            st.success("🎉 Excellent Performance!")
            st.balloons()
        elif total_score >= 25:
            st.warning("👍 Good Job! Keep practicing.")
        else:
            st.error("📚 Need more practice. Review the model answers.")

        # PDF Download button
        pdf_bytes = create_pdf(results_df, st.session_state.student_name, st.session_state.student_regno, total_score)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{st.session_state.student_regno}_result.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        if st.button("Take Another Exam", use_container_width=True):
            st.session_state.exam_started = False
            st.session_state.exam_submitted = False
            st.session_state.questions_selected = False
            st.session_state.selected_questions = pd.DataFrame()
            st.session_state.student_answers = {}
            st.rerun()
