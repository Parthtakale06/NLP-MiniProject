import streamlit as st
from PyPDF2 import PdfReader
# --- CHANGE 1: Import the Google Generative AI model ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# --- 1. SET UP THE ENVIRONMENT ---
load_dotenv()
# --- CHANGE 2: Look for the Google API Key ---
api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. DEFINE THE LANGCHAIN CHAINS (LCEL Syntax) ---

# Check for API Key before initializing LLM
if api_key:
    # --- CHANGE 3: Initialize the Gemini model ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.7)
else:
    st.error("GOOGLE_API_KEY not found. Please create a .env file and add your key.")
    st.stop()
    
# Output parser to convert chat messages to a string
output_parser = StrOutputParser()

# The prompts and chains remain the same, showcasing LangChain's flexibility.
# Prompt Template for generating interview questions
question_prompt_template = PromptTemplate.from_template(
    """
    You are an expert technical hiring manager reviewing a candidate for a specific role.
    Based on the provided job description and the candidate's resume, generate 5 to 7 insightful interview questions.
    The questions should cover a mix of technical skills, behavioral aspects, and project-specific inquiries.
    ---
    JOB DESCRIPTION:
    {jd}
    ---
    CANDIDATE'S RESUME:
    {resume}
    ---
    Generated Interview Questions (one question per line):
    """
)

# Chain for question generation
question_chain = question_prompt_template | llm | output_parser

# Prompt Template for generating model answers
answer_prompt_template = PromptTemplate.from_template(
    """
    You are a world-class career coach. Given an interview question, the candidate's resume, and the job description,
    formulate a strong model answer using the STAR method (Situation, Task, Action, Result).
    ---
    INTERVIEW QUESTION:
    {question}
    ---
    CANDIDATE'S RESUME:
    {resume}
    ---
    JOB DESCRIPTION:
    {jd}
    ---
    Model Answer (using STAR method):
    """
)

# Chain for answer generation
answer_chain = answer_prompt_template | llm | output_parser


# --- 3. BUILD THE STREAMLIT UI ---

st.set_page_config(page_title="Interview Prep Assistant", page_icon="🤖")
st.title("📄 Resume-to-Interview Prep Assistant")
st.markdown("Upload your resume and the job description to get personalized interview questions and model answers.")

jd = st.text_area("Paste the Job Description here:", height=250)
uploaded_file = st.file_uploader("Upload your Resume (PDF format only)", type="pdf")
submit_button = st.button("Generate Interview Prep")

# --- 4. PROCESSING AND OUTPUT ---

if submit_button:
    if uploaded_file is not None and jd:
        try:
            pdf_reader = PdfReader(uploaded_file)
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ""

            if not resume_text.strip():
                st.error("Could not extract text from the resume. Please try another PDF.")
            else:
                with st.spinner('Analyzing your documents...'):
                    generated_questions = question_chain.invoke({'jd': jd, 'resume': resume_text})
                    questions_list = generated_questions.strip().split('\n')
                    
                    st.subheader("🎉 Your Personalized Interview Questions & Answers")
                    st.markdown("---")

                    for question in questions_list:
                        question = question.strip()
                        if question:
                            st.markdown(f"#### ❓ {question.split('. ', 1)[-1]}")

                            with st.spinner("Crafting a model answer..."):
                                generated_answer = answer_chain.invoke({
                                    'question': question,
                                    'resume': resume_text,
                                    'jd': jd
                                })
                                with st.expander("Show Model Answer (STAR Method)"):
                                    st.markdown(generated_answer)
                            st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("NOTE: The Gemini API can sometimes be sensitive. If you see a 'Finish Reason: SAFETY' error, it might be due to the content of the resume or job description. Adjusting the prompt slightly can often help.")
    else:
        st.warning("Please upload your resume and paste the job description to proceed.")