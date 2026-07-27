import asyncio
from src.graph.workflow import WorkflowBuilder
from src.graph.state import GraphState

async def test_graph():
    builder = WorkflowBuilder()
    graph = builder.build_graph()
    
    initial_state = GraphState(
        resume_id="test_id_123",
        raw_resume="Software Engineer with Python and AWS",
        raw_jd="Looking for Python and AWS",
        parsed_resume=None,
        parsed_jd=None,
        gap_analyses=[],
        overall_fit_score=0.0,
        errors=[],
        skip_indexing=False
    )
    
    try:
        final = await graph.ainvoke(initial_state)
        print(final)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graph())
