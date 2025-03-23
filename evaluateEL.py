# agent for content evaluation - E-learning part only

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


# Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Cleaning steps
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", " ", text)
    text = text.strip()

    # Convert to JSON format without quotes around the text
    extracted_text = json.dumps(text)

    return extracted_text


def evaluate_elearning_script(
    state: DocumentEvaluationState,
) -> DocumentEvaluationState:
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["extracted_text"],
        template="""
        You are an evaluator tasked with verifying the structure of a e-learning scripts document. 
        Assess the presence and quality of the following key sections, assigning 1 if the criterion is met and 0 otherwise.
        Note that the criteria may not be explicitly mentioned as headings or text; however, an idea reflecting its presence will be present:
        
        ### Criteria to consider 
        1. Title & Learning Objectives
        -> Does the document include a Program Overview / Introduction?
            Are the learning goals explicitly stated and aligned with the content?   

        2. Scripted Narration & Dialogue
        -> Does the script include voice-over lines or narration?
        Are there clear trainer/learner conversations or dialogues included?
        
        3. Scene Descriptions
        -> Does the script describe visual elements, transitions, or on-screen text cues?
        
        4. Training Methodology Explanation
        -> Does it outline the Training Methodology / Approach?
        Are the scene descriptions detailed enough to guide the visual design?
            
        5. Course Outline & Content
        -> Does it include a Detailed Course Outline / Curriculum?
        
        6. Assessment & Evaluation Approac
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
            "Title & Learning Objectives": "(score)",
            "Scripted Narration & Dialogue": "(score)",
            "Scene Descriptions": "(score)",
            "Training Methodology Explanation": "(score)",
            "Course Outline & Content": "(score)",
            "Assessment & Evaluation Approach": "(score)",
            "Duration & Delivery Format": "(score)",
            "Trainer/Facilitator Profile": "(score)",
            "Pricing & Investment Details": "(score)",
            "Call to Action & Next Steps": "(score)"
        }},
        "Reasoning": {{
            "Title & Learning Objectives": "(reasoning)",
            "Scripted Narration & Dialogue": "(reasoning)",
            "Scene Descriptions": "(reasoning)",
            "Training Methodology Explanation": "(reasoning)",
            "Course Outline & Content": "(reasoning)",
            "Assessment & Evaluation Approach": "(reasoning)",
            "Duration & Delivery Format": "(reasoning)",
            "Trainer/Facilitator Profile": "(reasoning)",
            "Pricing & Investment Details": "(reasoning)",
            "Call to Action & Next Steps": "(reasoning)"
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

    return state.model_copy(update={"evaluation": evaluation})


# Initialize the StateGraph with the Pydantic schema
graph = StateGraph(state_schema=DocumentEvaluationState)
graph.add_node("Document_Evaluation", evaluate_elearning_script)
graph.add_edge(START, "Document_Evaluation")
graph.add_edge("Document_Evaluation", END)

compiled_graph = graph.compile()


# streamlit UI
def main():
    st.title("E-learning Script Evaluation Workflow")
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
