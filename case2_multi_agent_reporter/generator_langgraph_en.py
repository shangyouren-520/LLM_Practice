"""
=============================================================================
Case 2 (Advanced): Multi-Agent Financial Research Report System Based on
LangGraph StateGraph
-----------------------------------------------------------------------------
[Teaching Objective]:
This module implements the "Intelligent Analysis Platform V2 Integrated
Architecture" introduced on Slides 58-59. It upgrades the linear orchestration
workflow to a graph network centered on shared state by using LangGraph's
official StateGraph state-machine abstraction.

By comparing generator.py (V1 linear workflow) with generator_langgraph.py
(V2 graph-based state machine), learners can develop a deeper understanding of
modern multi-agent state-transition mechanisms used in production systems.
=============================================================================
"""

import os
import logging
import concurrent.futures
from typing import Dict, List, Any, TypedDict

# Import lower-level components.
from llm_factory_en import LLMFactory
from tools_en import DataTools
from agents_en import PlannerAgent, DataEngineerAgent, AnalystAgent
from generator_en import ResearchManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# 1. Define the shared multi-agent state (ReportState)
#    Corresponds to the central state ring on Slide 58.
# -----------------------------------------------------------------------------
class ReportState(TypedDict):
    """
    Global shared state board for financial research report generation.
    """

    stock_name: str
    stock_code: str
    sections: List[str]
    data_packet: Dict[str, Any]
    word_sections: List[Dict[str, str]]
    docx_path: str
    md_path: str


# -----------------------------------------------------------------------------
# 2. Define StateGraph nodes.
# -----------------------------------------------------------------------------
def planner_node(state: ReportState) -> Dict[str, Any]:
    """Node 1: Planner node."""
    logging.info("🌐 [LangGraph Node] >> Executing PlannerNode to define the report outline...")
    factory = LLMFactory()
    planner = PlannerAgent(llm_factory=factory)
    sections = planner.run({
        "input": f"Plan the core section structure for the {state['stock_name']} research report."
    })
    return {"sections": sections}


def data_engineer_node(state: ReportState) -> Dict[str, Any]:
    """Node 2: Data engineer node."""
    logging.info(
        "🌐 [LangGraph Node] >> Executing DataEngineerNode to retrieve data "
        "and generate charts..."
    )
    data_agent = DataEngineerAgent()
    data_packet = data_agent.run(
        stock_code=state["stock_code"],
        stock_name=state["stock_name"]
    )
    return {"data_packet": data_packet}


def analyst_node(state: ReportState) -> Dict[str, Any]:
    """Node 3: Analyst node that writes report sections in parallel."""
    logging.info(
        "🌐 [LangGraph Node] >> Executing AnalystNode to write report sections in parallel..."
    )
    factory = LLMFactory()
    analyst = AnalystAgent(llm_factory=factory)
    sections = state["sections"]
    data_context = state['data_packet']['data_context']

    word_sections = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sections)) as executor:
        future_map = {
            executor.submit(
                analyst.run,
                {
                    "input": (
                        f"Section title: {sec_title}\n"
                        f"Data context: {data_context}"
                    )
                }
            ): sec_title
            for sec_title in sections
        }
        for future in concurrent.futures.as_completed(future_map):
            sec_title = future_map[future]
            try:
                content = future.result()
                word_sections.append({"title": sec_title, "content": content})
                logging.info(f"  ✅ Section \"{sec_title}\" completed")
            except Exception as e:
                logging.error(f"  ❌ Failed to write section \"{sec_title}\": {e}")
                word_sections.append({
                    "title": sec_title,
                    "content": f"[Section generation failed] {e}"
                })

    return {"word_sections": word_sections}


def assembler_node(state: ReportState) -> Dict[str, Any]:
    """Node 4: Assembly and formatting node."""
    logging.info(
        "🌐 [LangGraph Node] >> Executing AssemblerNode to build the Word "
        "and Markdown deliverables..."
    )
    manager = ResearchManager()
    docx_path = manager._save_docx(
        stock_name=state["stock_name"],
        stock_code=state["stock_code"],
        data_packet=state["data_packet"],
        word_sections=state["word_sections"]
    )
    md_path = manager._save_markdown(
        stock_name=state["stock_name"],
        stock_code=state["stock_code"],
        data_packet=state["data_packet"],
        word_sections=state["word_sections"]
    )
    return {"docx_path": docx_path, "md_path": md_path}


# -----------------------------------------------------------------------------
# 3. Build and compile the LangGraph StateGraph workflow.
# -----------------------------------------------------------------------------
def build_report_graph():
    """Build the LangGraph state-flow graph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logging.warning(
            "⚠️ The langgraph package is not installed. "
            "Install it with: pip install langgraph"
        )
        return None

    # Initialize a graph based on ReportState.
    workflow = StateGraph(ReportState)

    # Register the four functional nodes.
    workflow.add_node("planner", planner_node)
    workflow.add_node("data_engineer", data_engineer_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("assembler", assembler_node)

    # Connect state edges to define the topological data flow.
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "data_engineer")
    workflow.add_edge("data_engineer", "analyst")
    workflow.add_edge("analyst", "assembler")
    workflow.add_edge("assembler", END)

    # Compile the StateGraph application.
    app = workflow.compile()
    return app


# -----------------------------------------------------------------------------
# Execution entry point.
# -----------------------------------------------------------------------------
def run_langgraph_workflow(stock_name: str = "CATL", stock_code: str = "300750"):
    graph = build_report_graph()

    initial_state = {
        "stock_name": stock_name,
        "stock_code": stock_code,
        "sections": [],
        "data_packet": {},
        "word_sections": [],
        "docx_path": "",
        "md_path": ""
    }

    if graph is not None:
        print("\n" + "=" * 60)
        logging.info("🚀 Starting the LangGraph multi-agent state-machine engine...")
        print("=" * 60)
        final_state = graph.invoke(initial_state)
        logging.info(
            f"🎉 LangGraph research report generation completed! "
            f"Word: {final_state['docx_path']}"
        )
        return final_state
    else:
        logging.info("ℹ️ Falling back to the standard Python ResearchManager workflow...")
        manager = ResearchManager()
        return manager.generate_report(stock_name, stock_code)


if __name__ == "__main__":
    run_langgraph_workflow("CATL", "300750")
