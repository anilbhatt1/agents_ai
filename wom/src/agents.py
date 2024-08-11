import os
from textwrap import dedent
from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from langchain.agents import load_tools
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

class PostRankingAgents:
    def __init__(self):
        self.api_file_path = 'api_keys.json'
        with open(json_file_path, 'r') as file:
            self.api_keys = json.load(file)        
        self.openai_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", 
                                       temperature=0.7,
                                       openai_api_key=self.api_keys['openai'])
        self.openai_gpt4 = ChatOpenAI(model_name="gpt-4", 
                                      temperature=0.7,
                                      openai_api_key=self.api_keys['openai'])        
        self.llama31_8b = Ollama(model='llama3.1:8b',
                                 openai_api_key=self.api_keys['ollama'])
    
    def agent_1_name(self):
        return Agent(
            role="Define agent 1 role here",
            backstory=dedent(f"""Define agent 1 backstory here"""),
            goal=dedent(f"""Define agent 1 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def agent_2_name(self):
        return Agent(
            role="Define agent 2 role here",
            backstory=dedent(f"""Define agent 2 backstory here"""),
            goal=dedent(f"""Define agent 2 goal here"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )    