import os
import uuid
from dotenv import load_dotenv
from src.graph.workflow import WorkflowBuilder

def run_verification():
    # Load environment variables (Gemini API Key, LangSmith config)
    load_dotenv()
    
    # 1. Dummy Inputs
    dummy_resume = """
    John Doe - Backend Developer
    Experience:
    - Built scalable REST APIs using Python and Django.
    - Managed relational databases using PostgreSQL.
    - Deployed web applications to AWS EC2 instances.
    - Led a team of 3 junior developers for the Q3 product launch.
    Skills: Python, Django, SQL, PostgreSQL, AWS, Leadership.
    Projects:
    - Built a real-time chat application using WebSockets and Redis.
    """
    
    dummy_jd = """
    Job Title: Senior Backend Engineer
    Responsibilities:
    - Design and build highly scalable microservices.
    - Mentor junior engineering staff and lead technical initiatives.
    Requirements:
    - Must have deep expertise in Python and FastAPI.
    - Strong knowledge of containerization (Kubernetes and Docker).
    - Experience with AWS (EC2, S3) is strictly required.
    - Nice to have: Experience with CI/CD pipelines and GitHub Actions.
    """
    
    # 2. Initialize the Graph
    print("Initializing Workflow Builder...")
    builder = WorkflowBuilder()
    graph = builder.build_graph()
    
    # 3. Define the Initial State
    initial_state = {
        "raw_resume": dummy_resume,
        "raw_jd": dummy_jd,
        "resume_id": str(uuid.uuid4()),
        "parsed_resume": None,
        "parsed_jd": None,
        "gap_analyses": [],
        "overall_fit_score": 0.0,
        "retry_count": 0,
        "is_grounded": False,
        "errors": [],
        "session_id": "test-session-123"
    }
    
    # 4. Execute the Graph
    print("\n--- STARTING LANGGRAPH EXECUTION ---")
    import asyncio
    # LangGraph ainvoke returns the final state dictionary
    final_state = asyncio.run(graph.ainvoke(initial_state))
    
    # 5. Output Results
    print("\n--- FINAL GAP ANALYSIS ---")
    print(f"Overall Fit Score: {final_state['overall_fit_score']}/10")
    
    if final_state["errors"]:
        print("\nERRORS DETECTED:")
        for err in final_state["errors"]:
            print(f"- {err}")
            
    print("\nDetailed Breakdown:")
    for analysis in final_state["gap_analyses"]:
        print(f"\n[Requirement]: {analysis.requirement}")
        print(f"[Score]: {analysis.match_score}/10")
        print(f"[Reasoning]: {analysis.reasoning}")
        print(f"[Gap]: {analysis.gap_description}")
        print("[Tailored Bullets]:")
        for bullet in analysis.tailored_bullets:
            print(f"  * {bullet}")

if __name__ == "__main__":
    run_verification()
