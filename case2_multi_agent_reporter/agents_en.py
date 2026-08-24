"""
=============================================================================
Case 2: Multi-Agent Multimodal Financial Research Report System - Agents
(Agent Role Definition Module)
-----------------------------------------------------------------------------
[Teaching Objective]:
This module implements the agent role definitions introduced on Slides 46-47:
1. AgentBase: abstract base class that encapsulates LLMFactory interaction and prompt composition;
2. PlannerAgent: plans the high-level report structure and returns a strict JSON list of section titles;
3. DataEngineerAgent: functional execution agent that retrieves data and generates trend charts accurately;
4. AnalystAgent: receives a target section and data context, then produces rigorous, detailed analysis.
=============================================================================
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional

from llm_factory_en import LLMFactory
from tools_en import DataTools


class AgentBase:
    """Base class for all agents."""

    def __init__(self, name: str, role_prompt: str, llm_factory: Optional[LLMFactory] = None):
        self.name = name
        self.role_prompt = role_prompt
        self.llm_factory = llm_factory or LLMFactory()
        self.llm = self.llm_factory.get_llm()

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM and remove <think> tags and redundant control characters."""
        try:
            if hasattr(self.llm, "invoke"):
                res = self.llm.invoke(prompt)
                res_text = res.content if hasattr(res, "content") else str(res)
            else:
                res_text = str(self.llm(prompt))
        except Exception as e:
            logging.error(f"❌ Agent [{self.name}] model invocation failed: {e}")
            res_text = ""

        # Remove DeepSeek-R1 <think> chain-of-thought tags and retain only the final generated text.
        cleaned_text = re.sub(r'<think>.*?</think>', '', res_text, flags=re.DOTALL).strip()
        # Remove abnormally repeated punctuation or letter streams to guard against runaway token loops.
        cleaned_text = re.sub(r'([A-Za-z,.]{4,})\1{3,}', '', cleaned_text)
        return cleaned_text


# -----------------------------------------------------------------------------
# Agent 1: PlannerAgent (corresponds to the left side of Slide 46)
# -----------------------------------------------------------------------------
class PlannerAgent(AgentBase):
    """
    Planner agent responsible for defining the overall report outline and structure.
    """

    def __init__(self, llm_factory: Optional[LLMFactory] = None):
        super().__init__(
            name="Planner",
            role_prompt=(
                "You are the chief editor of a professional financial research report. "
                "Return only a JSON list containing exactly three core section titles, "
                "for example: [\"Section 1\", \"Section 2\", \"Section 3\"]. "
                "Do not include any Markdown syntax, commentary, or explanatory text."
            ),
            llm_factory=llm_factory
        )

    def run(self, input_dict: Dict[str, Any]) -> List[str]:
        prompt = f"{self.role_prompt}\n\nTask: {input_dict.get('input', '')}"
        logging.info(f"📋 [{self.name}Agent] Planning the research report structure...")
        raw_output = self._invoke_llm(prompt)

        # Structured extraction with defensive fault tolerance.
        try:
            # Extract the JSON array fragment.
            json_match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
            if json_match:
                sections = json.loads(json_match.group())
                if isinstance(sections, list) and len(sections) > 0:
                    logging.info(
                        f"✅ [{self.name}Agent] Successfully generated "
                        f"{len(sections)} core sections: {sections}"
                    )
                    return [str(s).strip() for s in sections]
        except Exception as e:
            logging.warning(
                f"⚠️ [{self.name}Agent] Failed to parse the LLM JSON output ({e}); "
                "using the standard instructional fallback outline."
            )

        # Standard instructional fallback outline (Slide 49).
        default_sections = ["Market Review", "Technical Analysis", "Trading Recommendations"]
        return default_sections


# -----------------------------------------------------------------------------
# Agent 2: DataEngineerAgent (corresponds to Slide 47)
# -----------------------------------------------------------------------------
class DataEngineerAgent:
    """
    Data engineer agent implemented as a functional agent.

    It does not perform subjective or creative reasoning. Instead, it strictly
    invokes DataTools according to business logic, bridging cognition and execution.
    """

    def __init__(self):
        self.name = "DataEngineer"

    def run(self, stock_code: str, stock_name: str) -> Dict[str, Any]:
        logging.info(f"📊 [{self.name}Agent] Retrieving data and generating charts...")

        # 1. Invoke the tool to obtain structured trading data.
        df = DataTools.get_stock_data(symbol=stock_code, stock_name=stock_name, days=60)

        # 2. Invoke the tool to generate a professional trend chart.
        chart_title = f"{stock_name} ({stock_code}) Price Trend"
        chart_filename = f"{stock_code}_trend.png"
        chart_path = DataTools.draw_chart(df, chart_title, chart_filename)

        # 3. Extract key statistics and package them into analysis context.
        current_price = df["Close"].iloc[-1]
        start_price = df["Close"].iloc[0]
        change_pct = round(((current_price - start_price) / start_price) * 100, 2)

        data_context = (
            f"Current price: CNY {current_price}; "
            f"60-day change: {change_pct}%; "
            f"period high: CNY {df['High'].max()}; "
            f"period low: CNY {df['Low'].min()}"
        )

        logging.info(f"✅ [{self.name}Agent] Data package completed: {data_context}")

        return {
            "df": df,
            "chart_path": chart_path,
            "current_price": current_price,
            "change_pct": change_pct,
            "data_context": data_context
        }


# -----------------------------------------------------------------------------
# Agent 3: AnalystAgent (corresponds to the right side of Slide 46)
# -----------------------------------------------------------------------------
class AnalystAgent(AgentBase):
    """
    Analyst agent that writes focused, professional analysis based on a specified
    section title and the supplied market-data context.
    """

    def __init__(self, llm_factory: Optional[LLMFactory] = None):
        super().__init__(
            name="Analyst",
            role_prompt=(
                "You are a senior financial analyst. Based on the supplied market-data context, "
                "write a clear, rigorous, and professional analysis of approximately 150 words "
                "for the specified section. Output the analysis paragraph directly, without any "
                "Markdown headings or irrelevant prefixes."
            ),
            llm_factory=llm_factory
        )

    def run(self, input_dict: Dict[str, Any]) -> str:
        prompt = (
            f"{self.role_prompt}\n\n"
            f"[Analysis Task and Context]:\n{input_dict.get('input', '')}\n\n"
            "Output the professional analysis paragraph directly:"
        )
        content = self._invoke_llm(prompt)
        # Remove any accidentally generated Markdown headings.
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE).strip()
        if not content:
            content = (
                "Based on recent market performance and price-volume dynamics, the security's "
                "overall behavior remains broadly aligned with its industry fundamentals. Investors "
                "should closely monitor volume confirmation and the performance of key support levels."
            )
        return content
