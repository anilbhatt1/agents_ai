import os
from textwrap import dedent
from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from langchain.agents import load_tools

from langchain.llms import Ollama

class PostRankingAgents:
    def __init__(self):
        self.llm_llama31_8b = Ollama(model=os.environ['MODEL'])