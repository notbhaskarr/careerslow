import asyncio
import os
import uuid
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langsmith import Client, evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.workflow import WorkflowBuilder
from src.graph.state import GraphState

# 1. Initialize LangSmith Client
client = Client()

# The name of the dataset you will create in the LangSmith UI
DATASET_NAME = "careerslow_eval_set"

# 2. Define the Target Pipeline (Wrapper around our LangGraph)
async def pipeline_wrapper(inputs: dict) -> dict:
    """
    Wraps the LangGraph workflow to be callable by LangSmith's evaluate().
    Expects 'raw_resume' and 'raw_jd' in inputs.
    """
    builder = WorkflowBuilder()
    graph = builder.build_graph()
    
    # Create initial state
    initial_state = GraphState(
        raw_resume=inputs.get("raw_resume", ""),
        raw_jd=inputs.get("raw_jd", ""),
        resume_id=f"eval_res_{uuid.uuid4().hex[:8]}",
        parsed_resume=None,
        parsed_jd=None,
        gap_analyses=[],
        overall_fit_score=0.0,
        score_breakdown={},
        errors=[],
        skip_indexing=False,
        retry_count=0,
        is_grounded=True,
        session_id="eval-session",
    )
    
    try:
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        # We return the gap_analyses as a structured dictionary for the evaluators
        return {
            "overall_fit_score": final_state["overall_fit_score"],
            "gap_analyses": [
                {
                    "requirement": gap.requirement,
                    "match_score": gap.match_score,
                    "gap_description": gap.gap_description,
                    "tailored_bullets": gap.tailored_bullets
                }
                for gap in final_state["gap_analyses"]
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# 3. Define Evaluators (LLM-as-a-Judge)

def jargon_evaluator(run, example) -> dict:
    """
    Evaluates whether the tailored bullets contain corporate jargon.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
    
    # Extract the generated output
    output = run.outputs
    if "error" in output:
        return {"key": "jargon_free", "score": 0, "comment": "Pipeline failed."}
        
    analyses = output.get("gap_analyses", [])
    
    # Collect all bullets
    all_bullets = []
    for a in analyses:
        all_bullets.extend(a.get("tailored_bullets", []))
        
    if not all_bullets:
        return {"key": "jargon_free", "score": 1, "comment": "No bullets generated, but technically jargon free."}
        
    bullets_text = "\n".join(f"- {b}" for b in all_bullets)
    
    prompt = (
        "You are a strict technical editor. Read the following resume bullets.\n"
        "Are there any meaningless corporate buzzwords or fluff? (e.g., 'Leveraged', 'Spearheaded', 'Synergized', 'Orchestrated').\n"
        "Return a JSON object with 'score' (1 if completely jargon-free, 0 if jargon is present) and 'reasoning' (string).\n\n"
        f"Bullets:\n{bullets_text}"
    )
    
    # Quick structured invocation
    try:
        # We'll use a standard generation and parse JSON
        response = llm.invoke(prompt)
        text = response.content.strip().strip('`').strip('json')
        result = json.loads(text)
        return {"key": "jargon_free", "score": int(result.get("score", 0)), "comment": result.get("reasoning", "")}
    except Exception as e:
        return {"key": "jargon_free", "score": 0, "comment": f"Eval failed: {str(e)}"}

def groundedness_evaluator(run, example) -> dict:
    """
    Evaluates whether the tailored bullets hallucinated any skills not present in the resume.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
    
    # Extract inputs and outputs
    raw_resume = example.inputs.get("raw_resume", "")
    output = run.outputs
    
    if "error" in output:
        return {"key": "is_grounded", "score": 0, "comment": "Pipeline failed."}
        
    analyses = output.get("gap_analyses", [])
    all_bullets = []
    for a in analyses:
        all_bullets.extend(a.get("tailored_bullets", []))
        
    if not all_bullets:
        return {"key": "is_grounded", "score": 1, "comment": "No bullets to hallucinate."}
        
    bullets_text = "\n".join(f"- {b}" for b in all_bullets)
    
    prompt = (
        "You are a strict technical auditor. Compare the generated resume bullets against the original candidate resume.\n"
        "Did the generated bullets invent any skills, technologies, or specific project experiences that are completely absent from the original resume?\n"
        "Return a JSON object with 'score' (1 if strictly grounded, 0 if it hallucinated new skills) and 'reasoning' (string).\n\n"
        f"--- ORIGINAL RESUME ---\n{raw_resume}\n\n"
        f"--- GENERATED BULLETS ---\n{bullets_text}"
    )
    
    try:
        response = llm.invoke(prompt)
        text = response.content.strip().strip('`').strip('json')
        result = json.loads(text)
        return {"key": "is_grounded", "score": int(result.get("score", 0)), "comment": result.get("reasoning", "")}
    except Exception as e:
        return {"key": "is_grounded", "score": 0, "comment": f"Eval failed: {str(e)}"}

from langsmith import aevaluate

async def main():
    print(f"Starting LangSmith Evaluation for dataset: {DATASET_NAME}")
    try:
        results = await aevaluate(
            pipeline_wrapper,
            data=DATASET_NAME,
            evaluators=[jargon_evaluator, groundedness_evaluator],
            experiment_prefix="CareersLow-Pipeline-Eval",
            description="Testing the Decomposed Generative Pipeline for Jargon and Groundedness",
        )
        print("\nEvaluation kicked off successfully! View the results in the LangSmith UI.")
    except Exception as e:
        print(f"\nError running evaluation: {str(e)}")
        print(f"Did you create a dataset named '{DATASET_NAME}' in the LangSmith UI?")

# 4. Run Evaluation
if __name__ == "__main__":
    asyncio.run(main())
