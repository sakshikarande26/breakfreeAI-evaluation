# Routing Agent - Training Proposal or E-learning Script

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import re
from langchain_core.messages import HumanMessage
import json

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM with retries
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, max_retries=5)


# Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    # Cleaning steps
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", " ", text)
    extracted_text = text.strip()

    return extracted_text


def classify_with_llm(extracted_text):
    prompt = f"""
    You are an AI-powered document classification and compliance agent. Your task is to analyze the structure of a given PDF document and determine whether it follows the format of either:

    - Training Proposal
    - E-learning Script

    Evaluation Criteria:
    - Check for the presence of each required structural element.
    - Assign 1 point if present, 0 if absent.
    - Compute total scores out of 10 for both categories.
    - If a category scores 8 or above, it is compliant; otherwise, it is non-compliant.
    - Classify the document based on the higher score (or best fit if tied).

    Training Proposal Criteria (Score out of 10):
    1. Program Overview / Introduction
    2. Clearly Defined Learning Objectives
    3. Target Audience Description
    4. Training Methodology Explanation
    5. Course Outline & Content
    6. Assessment & Evaluation Approach
    7. Duration & Delivery Format
    8. Trainer/Facilitator Profile
    9. Pricing & Investment Details
    10. Call to Action / Next Steps

    E-learning Script Criteria (Score out of 10):
    1. Title & Learning Objectives
    2. Scripted Narration & Dialogue
    3. Scene Descriptions
    4. Interactivity Cues (quizzes, branching, activities)
    5. Instructional Flow
    6. Speaker Labels (narrator, trainer, learner)
    7. On-Screen Text vs. Narration Distinction
    8. Interactivity Instructions
    9. Formatting Consistency
    10. File Format Compliance

    Document for Analysis:
    {extracted_text}

    Return output strictly in JSON format with no extra text. 

    Example Output:
    {{
        "Document Name": "Extracted_Name",
        "Total Pages": 1,
        "Classification": {{
            "Training Proposal Score": "(total score)",
            "E-learning Script Score": "(total score)",
            "Predicted Category": "(Category)",
            "Reasoning": "(Reasoning)",
            "Confidence": "(High/Medium/Low)",
            "Compliance": "(Compliant/Non-Compliant)"
        }},
        "Missing Sections": {{
            "Training Proposal": "(missing parameters for training proposal)",
            "E-learning Script": "(missing parameters for e-learning script)"
        }}
    }}
    """

    try:
        response = llm([HumanMessage(content=prompt)])
        if not response or not response.content.strip():
            return None

        # Clean the LLM response (remove markdown formatting if present)
        cleaned_response = response.content.strip()
        cleaned_response = re.sub(r"^```json\n", "", cleaned_response)
        cleaned_response = re.sub(r"```$", "", cleaned_response)

        return cleaned_response
    except Exception as e:
        print("Error in LLM call:", str(e))
        return None


def main():
    st.title("Document Classification and Compliance Checker")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        st.subheader("Extracted Text")
        st.markdown(extracted_text)

        if st.button("Classify Document"):
            with st.spinner("Classifying document..."):
                classification_result = classify_with_llm(extracted_text)

            if classification_result:
                try:
                    # Clean and validate JSON before parsing
                    cleaned_json = classification_result.strip()
                    cleaned_json = re.sub(
                        r"^```json\n", "", cleaned_json
                    )  # Remove leading ```json
                    cleaned_json = re.sub(
                        r"```$", "", cleaned_json
                    )  # Remove trailing ```

                    # Attempt to parse JSON
                    parsed_result = json.loads(cleaned_json)

                    st.subheader("Classification Result")
                    st.json(parsed_result)

                except json.JSONDecodeError as e:
                    st.error(f"JSON Parsing Error: {e}")
                    st.write("Raw Response from LLM:")
                    st.code(
                        classification_result, language="json"
                    )  # Display raw output for debugging

            else:
                st.error("Failed to classify the document. Please try again.")


if __name__ == "__main__":
    main()
