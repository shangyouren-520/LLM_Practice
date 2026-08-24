"""
=============================================================================
Case 2: Multi-Agent Multimodal Financial Research Report System - LLMFactory
(Model Factory Module)
-----------------------------------------------------------------------------
[Teaching Objective]:
This module implements the core LLMFactory design introduced on Slide 44:
1. Prefer the free cloud-hosted Qwen/Qwen3-8B model on SiliconFlow for
   millisecond-level inference with zero local GPU-memory usage;
2. Support local Ollama models (deepseek-r1:1.5b / 7b) with keep_alive="5m"
   to keep the model resident in GPU memory;
3. Provide a fully offline Mock LLM fallback so the workflow remains functional
   even when no network connection or local Ollama service is available.
=============================================================================
"""

import os
import logging
from typing import Any, Optional

# Load the API key and model configuration from .env.
# Prefer DOTENV_PATH supplied by the .bat launcher, then search the local/runtime
# directory hierarchy as a fallback.
try:
    from dotenv import load_dotenv

    _env_candidates = []
    if os.getenv("DOTENV_PATH"):
        _env_candidates.append(os.getenv("DOTENV_PATH"))

    _here = os.path.dirname(os.path.abspath(__file__))
    _env_candidates.append(os.path.join(_here, ".env"))
    _env_candidates.append(os.path.join(os.path.dirname(_here), "runtime", ".env"))
    _env_candidates.append(os.path.join(os.path.dirname(_here), ".env"))

    for _p in _env_candidates:
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# SiliconFlow cloud API configuration.
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
ONLINE_LLM_MODEL = os.getenv("ONLINE_LLM_MODEL", "Qwen/Qwen3-8B")

# Default local Ollama configuration.
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "deepseek-r1:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class MockLLM:
    """
    Mock LLM fallback for development and instruction.

    It activates automatically when the system is fully offline and no local
    Ollama service is available, returning representative financial-research
    outputs for demonstration purposes.
    """

    def __init__(self):
        self.model_name = "mock-qwen-financial"

    def invoke(self, prompt_input: Any) -> str:
        prompt_text = str(prompt_input)

        # Planner response.
        if "chief editor of a professional financial research report" in prompt_text or \
                "JSON list" in prompt_text:
            return (
                '["Industry Background and Market Position", '
                '"In-Depth Financial and Technical Trend Analysis", '
                '"Outlook and Investment Recommendations"]'
            )

        # Analyst response.
        if "senior financial analyst" in prompt_text or "Data context" in prompt_text:
            return (
                "Recent market performance and price-volume dynamics indicate that the security "
                "has shown clear signs of stabilization and recovery after a period of consolidation. "
                "From a fundamental perspective, continued growth in global new-energy and power-battery "
                "deployment supports the company's technological barriers and scale advantages. "
                "Investors should closely monitor moving-average support and volume confirmation, "
                "while maintaining a constructive medium- to long-term view."
            )

        return (
            "[System Demonstration Analysis]: Based on the supplied data context, "
            "the core operating indicators remain stable and demonstrate solid resilience to risk."
        )


class LLMFactory:
    """
    Model factory responsible for creating LLM instances and handling
    connection-level failover.
    """

    def __init__(
        self,
        model_name: str = ONLINE_LLM_MODEL,
        temperature: float = 0.6
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.active_llm = None
        self._init_llm()

    def _init_llm(self):
        """
        Prefer the free cloud-hosted Qwen-7B model, then try local Ollama,
        and finally fall back to MockLLM.
        """
        # 1. Try the free cloud-hosted Qwen model on SiliconFlow.
        if SILICONFLOW_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                logging.info(
                    f"🔄 Connecting to the free SiliconFlow cloud LLM "
                    f"({self.model_name})..."
                )
                test_llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=SILICONFLOW_API_KEY,
                    openai_api_base=SILICONFLOW_BASE_URL,
                    temperature=self.temperature,
                    max_tokens=400,
                    streaming=True,
                    extra_body={"enable_thinking": False}
                )
                _ = test_llm.invoke("hi")
                self.active_llm = test_llm
                logging.info(
                    f"✅ Successfully connected to the SiliconFlow cloud model: "
                    f"{self.model_name}!"
                )
                return
            except Exception as e:
                logging.warning(
                    f"⚠️ Cloud API connection failed ({e}); trying local Ollama..."
                )

        # 2. Try local Ollama.
        try:
            from langchain_ollama import ChatOllama

            logging.info(
                f"🔄 Attempting to connect to the local Ollama service "
                f"({OLLAMA_BASE_URL})..."
            )
            test_llm = ChatOllama(
                model=LOCAL_LLM_MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=self.temperature,
                keep_alive="5m"
            )
            _ = test_llm.invoke("hi")
            self.active_llm = test_llm
            logging.info("✅ Successfully connected to the local Ollama LLM service!")
            return
        except Exception as e:
            logging.warning(
                f"⚠️ Unable to connect to local Ollama ({e}). "
                "Automatically switching to the instructional Mock fallback mode."
            )

        # 3. Final fallback: MockLLM.
        self.active_llm = MockLLM()

    def get_llm(self) -> Any:
        """Return the currently available LLM instance."""
        return self.active_llm
