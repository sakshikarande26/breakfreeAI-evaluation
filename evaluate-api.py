#strutcure eval and classification + routing + quality evaluation
import pdfplumber
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import re
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import json
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

app = FastAPI()

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM with retries
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, max_retries=5)


class AgentState(BaseModel):
    file_path: bytes | None = None  
    extracted_text: str | None = None
    evaluation: dict | None = None
    classification: dict | None = None
    has_error: bool = False
    error_message: str | None = None
    predicted_category: str | None = None
    next: str | None = None  


@app.get("/")
async def root():
    return {"message": "Hello World"}

# Node: Content Extraction (Integrated PDF processing)
def extractContent(state: AgentState):
    # Convert bytes to BytesIO object for PDF processing
    pdf_file = BytesIO(state.file_path)
    
    # PDF extraction and cleaning
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    # Cleaning pipeline
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)  # Remove control chars
    text = re.sub(r"\n+", " ", text)  # Replace newlines
    text = re.sub(r"\t+", " ", text)  # Replace tabs
    extracted_text = text.strip()  # Final cleanup

    # Update state
    state.extracted_text = extracted_text
    return state


# Node: Classification with Enhanced Structure
def classifyDocument(state: AgentState):
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
    {state.extracted_text}

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
        # Generate and execute prompt
        response = llm([HumanMessage(content=prompt)])

        # Debug the response
        print(f"Raw response: {response.content}")

        if not response.content or not response.content.strip():
            raise ValueError("Empty LLM response")

        # Enhanced cleaning of the response content
        cleaned_content = response.content.strip()

        # Remove any markdown code block indicators
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split("```json", 1)[1]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.split("```", 1)[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit("```", 1)[0]

        cleaned_content = cleaned_content.strip()

        if not cleaned_content:
            raise ValueError("Empty JSON content after cleaning")

        # Parse the JSON
        result = json.loads(cleaned_content)

        # Store classification directly in state
        state.classification = result.get("Classification", {})
        state.predicted_category = state.classification.get("Predicted Category")

    except json.JSONDecodeError as e:
        state.has_error = True
        state.error_message = f"Classification failed: Invalid JSON response: {str(e)}\nResponse: {response.content[:100]}..."
    except Exception as e:
        state.has_error = True
        state.error_message = f"Classification failed: {str(e)}"

    return state


# Node: e-learning script Evaluation
def evaluateElearning(state: AgentState) -> AgentState:
    prompt_template = PromptTemplate(
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
        "Feedback":"(detailed feedback of missing sections)",
        "Suggestions":" (a numbered list of improvements needed in the documents to increase the total score for high quality)"
        }}
        """,
        input_variables=["extracted_text"],
    )

    try:
        # Generate and execute evaluation prompt
        prompt = prompt_template.format(extracted_text=state.extracted_text)
        response = llm([HumanMessage(content=prompt)])

        if not response.content:
            raise ValueError("Empty LLM response")

        # Clean the response content
        cleaned_content = response.content.strip()

        # Remove any markdown code block indicators
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split("```json", 1)[1]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.split("```", 1)[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit("```", 1)[0]

        cleaned_content = cleaned_content.strip()

        # Parse the JSON
        evaluation_data = json.loads(cleaned_content)

        # Store directly in state - use the standard evaluation field
        state.evaluation = evaluation_data
        return state  # Return the state object, not the response content

    except json.JSONDecodeError as e:
        state.has_error = True
        state.error_message = f"JSON parsing failed: {str(e)}"
    except Exception as e:
        state.has_error = True
        state.error_message = f"Evaluation failed: {str(e)}"

    return state  # Return state even if there's an error


def evaluateTrainingProposal(state: AgentState) -> AgentState:
    prompt_template = PromptTemplate(
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

        Generate output strictly in the following JSON format with no additional text or formatting:

        {{
          "scores": {{
            "Presence of Introduction & Program Overview": "1",
            "Clearly defined Learning Objectives": "1",
            "Target Audience Description": "1",
            "Training Methodology Explanation": "1",
            "Course Outline & Content": "1",
            "Assessment & Evaluation Approach": "1",
            "Duration & Delivery Format": "1",
            "Trainer/Facilitator Profile": "1",
            "Pricing & Investment Details": "1",
            "Call to Action & Next Steps": "1"
          }},
          "Reasoning": {{
            "Presence of Introduction & Program Overview": "reason here",
            "Clearly defined Learning Objectives": "reason here",
            "Target Audience Description": "reason here",
            "Training Methodology Explanation": "reason here",
            "Course Outline & Content": "reason here",
            "Assessment & Evaluation Approach": "reason here",
            "Duration & Delivery Format": "reason here",
            "Trainer/Facilitator Profile": "reason here",
            "Pricing & Investment Details": "reason here",
            "Call to Action & Next Steps": "reason here"
          }},
          "Total Score": "8",
          "Compliance Status": "Compliant",
          "Feedback": "detailed feedback here",
          "Suggestions": "1. First suggestion\\n2. Second suggestion"
        }}
        """,
        input_variables=["extracted_text"],
    )

    try:
        prompt = prompt_template.format(extracted_text=state.extracted_text)
        response = llm([HumanMessage(content=prompt)])

        if not response.content:
            raise ValueError("Empty LLM response")

        # Clean the response content
        cleaned_content = response.content.strip()

        # Remove any markdown code block indicators
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split("```json", 1)[1]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.split("```", 1)[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit("```", 1)[0]
                                                                                                                     
        cleaned_content = cleaned_content.strip()

        # Parse JSON with better error handling
        try:
            evaluation_data = json.loads(cleaned_content)
        except json.JSONDecodeError as je:
            raise ValueError(f"JSON parsing failed: {str(je)}\nResponse content: {cleaned_content[:200]}...")

        # Store in state
        state.evaluation = evaluation_data
        if not hasattr(state, "predicted_category"):
            state.predicted_category = "Training Proposal"
        return state

    except Exception as e:
        state.has_error = True
        state.error_message = f"Training Proposal Evaluation failed: {str(e)}"
        return state


def routing_agent(state: AgentState):
    if state.has_error:
        state.next = "error_handler"
        return state

    if not state.predicted_category:
        state.next = "unknown_category"
        return state

    category = state.predicted_category.replace(" ", "_").lower()
    if category in ["training_proposal", "e-learning_script"]:
        state.next = category
    else:
        state.next = "unknown_category"

    return state


def handle_error(state: AgentState):
    # Don't put Streamlit commands here - they won't work in the graph
    # Just update the state
    state.evaluation = {"error": True, "message": state.error_message}
    return state


def handle_unknown_category(state: AgentState):
    # Don't put Streamlit commands here - they won't work in the graph
    # Just update the state
    state.evaluation = {
        "unknown_category": True,
        "message": "The document category could not be determined.",
    }
    return state


graph = StateGraph(AgentState)

# Add nodes
graph.add_node("extract_text", extractContent)
graph.add_node("classify_document", classifyDocument)
graph.add_node("routing", routing_agent)
graph.add_node("training_proposal", evaluateTrainingProposal)
graph.add_node("elearning_script", evaluateElearning)
graph.add_node("error_handler", handle_error)
graph.add_node("unknown_category", handle_unknown_category)

# Define edges
graph.set_entry_point("extract_text")
graph.add_edge("extract_text", "classify_document")
graph.add_edge("classify_document", "routing")

# Conditional routing edges
graph.add_conditional_edges(
    "routing",
    lambda state: state.next,  # Use dot notation instead of state["next"]
    {
        "training_proposal": "training_proposal",
        "e-learning_script": "elearning_script",
        "error_handler": "error_handler",
        "unknown_category": "unknown_category",
    },
)

# Endings for evaluation nodes
graph.add_edge("training_proposal", END)
graph.add_edge("elearning_script", END)
graph.add_edge("error_handler", END)
graph.add_edge("unknown_category", END)


# Compile graph
compiled_graph = graph.compile()

@app.post("/evaluate")
async def process_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a PDF file. The file is processed in-memory without saving it to disk.
    """
    try:
        # 1. Read the file content into memory
        file_content = await file.read()

        # 2. Create AgentState with the file path
        state = AgentState(file_path=file_content)

        # 3. Execute compiled LangGraph chain
        result = compiled_graph.invoke(state)

        # 4. Return results - access dictionary values directly
        return {
            "classification": result.get("classification"),
            "evaluation": result.get("evaluation"),
            "error": result.get("error_message") if result.get("has_error") else None
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
