# agent for content evaluation - Training Proposal part only

import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


class DocumentEvaluationState(BaseModel):
    extracted_text: str
    evaluation: dict | None = None


# initialise llm
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, max_retries=2)


# extract text from pdf
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        extracted_text = ""
        for page in pdf.pages:
            extracted_text += page.extract_text()

    extracted_text = re.sub(r"[\x00-\x1F\x7F]", "", extracted_text)
    extracted_text = json.dumps(extracted_text)

    return extracted_text


def evaluate_training_proposal(
    state: DocumentEvaluationState,
) -> DocumentEvaluationState:
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template="""
        You are an evaluator tasked with verifying the structure of a training proposal document. 
        Assess the presence and quality of the following key sections, assigning 1 if the criterion is met and 0 otherwise.
        Note that the criteria may not be explicitly mentioned as headings or text; however, an idea reflecting its presence will be present:
        
        ### Criteria to consider 
        1. Presence of Introduction & Program Overview
           -> Does the document include a Program Overview / Introduction?

        2. Clearly defined Learning Objectives
           -> Does it have a Learning Objectives / Expected Outcomes section?    
        
        3. Target Audience Description
           -> Does it have a Target Audience Description section?
        
        4. Training Methodology Explanation
           -> Does it outline the Training Methodology / Approach?
        
        5. Course Outline & Content
           -> Does it include a Detailed Course Outline / Curriculum?
        
        6. Assessment & Evaluation Approach
           -> Does it outline the Assessment & Evaluation Approach?
        
        7. Duration & Delivery Format
           -> Does it specify the Duration / Delivery Format (e.g., in-person, virtual, blended)?
        
        8. Trainer/Facilitator Profile
           -> Does it provide details on the Facilitator / Trainer Profile?
        
        9. Pricing & Investment Details
           -> Does it include Pricing / Investment Details?
        
        10. Call to Action & Next Steps
           -> Does it conclude with a Call to Action / Next Steps?

        If the document lacks these elements, flag the criteria as 0 and provide reasoning for the same. If it is explicitly or implicitly present, flag it as 1 and provide reasoning for the same.

        **Scoring System:**
        - Total Score: ___ / 10 
        - Pass Requirement: At least 8/10 (80%)
          - If ≥ 8/10, proceed to full evaluation.
          - If < 8/10, flag as non-compliant and provide feedback on missing sections. 

        **Document Content:**
        {extracted_text}

        Generate output strictly in the following JSON format: (Ensure you respond **only** with the JSON output, without any extra text, explanation, or formatting.
        Wrap your response like this)

        {{
          "content": "{extracted_text}",
          "scores": {{
            "Presence of Introduction & Program Overview": 1,
            "Clearly defined Learning Objectives": 0,
            "Target Audience Description": 1,
            "Training Methodology Explanation": 0,
            "Course Outline & Content": 1,
            "Assessment & Evaluation Approach": 0,
            "Duration & Delivery Format": 1,
            "Trainer/Facilitator Profile": 1,
            "Pricing & Investment Details": 0,
            "Call to Action & Next Steps": 1
          }},
          "Reasoning": {{
            "Presence of Introduction & Program Overview": "(reason)",
            "Clearly defined Learning Objectives": "(reason)",
            "Target Audience Description": "(reason)",
            "Training Methodology Explanation": "(reason)",
            "Course Outline & Content": "(reason)",
            "Assessment & Evaluation Approach": "(reason)",
            "Duration & Delivery Format": "(reason)",
            "Trainer/Facilitator Profile": "(reason)",
            "Pricing & Investment Details": "(reason)",
            "Call to Action & Next Steps": "(reason)"
          }},
          "Total Score": "(score)",
          "Compliance Status": "(Compliant/Non-Compliant)",
          "Feedback": "(summary of missing sections)"
        }}
        """,
    )

    prompt = prompt_template.format(extracted_text=state.extracted_text)
    response = llm.invoke(prompt)

    response_content = response.content

    # Parse the JSON response
    try:
        evaluation = json.loads(response_content)
    except json.JSONDecodeError as e:
        evaluation = {"error": f"Failed to parse LLM response. Error: {str(e)}"}

    state.evaluation = evaluation
    return state


# Initialize the StateGraph with the Pydantic schema
graph = StateGraph(state_schema=DocumentEvaluationState)
graph.add_node("Document_Evaluation", evaluate_training_proposal)
graph.add_edge(START, "Document_Evaluation")
graph.add_edge("Document_Evaluation", END)

compiled_graph = graph.compile()


# streamlit UI
def main():
    st.title("Training Proposal Evaluation Workflow")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        extracted_text = extract_text_from_pdf(uploaded_file)

        st.subheader("📜 Extracted Text")
        st.write(extracted_text)

        # Create an initial state instance
        initial_state = DocumentEvaluationState(extracted_text=extracted_text)

        # Invoke the compiled graph with the initial state
        final_state = compiled_graph.invoke(initial_state)
        evaluation = final_state.evaluation

        # Display results in JSON format
        st.subheader("📊 Evaluation Results")
        st.json(evaluation)


if __name__ == "__main__":
    main()
