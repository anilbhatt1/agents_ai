import os
import json
import yaml
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
        with open(self.api_file_path, 'r') as file:
            self.api_keys = json.load(file)        
        self.openai_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", 
                                       temperature=0.7,
                                       openai_api_key=self.api_keys['openai'])
        self.openai_gpt4 = ChatOpenAI(model_name="gpt-4", 
                                      temperature=0.7,
                                      openai_api_key=self.api_keys['openai'])        
        self.llama31_8b = Ollama(model='llama3.1:8b',
                                 openai_api_key=self.api_keys['ollama'])
        self.gemma2_9b = Ollama(model='gemma2:9b',
                                 openai_api_key=self.api_keys['ollama'])   
        self.cfg_file_path = 'casaai_config.yaml'

        with open(self.cfg_file_path, 'r') as yaml_file:
            self.cfg = yaml.safe_load(yaml_file)
        self.product_long = self.cfg.get('product_long_description', '')      
        self.product_short = self.cfg.get('product_short_description', '')     
    
    def content_analysis_agent(self):
        backstory = "You are a content analyst with expertise in analyzing web content and \
                     extracting relevant information. You are responsible for ensuring that \
                     content is relevant, high-quality, and aligned with the marketing of \
                    {self.product_short}. "
        return Agent(
            role="Content Analyst",
            goal="Analyze web content and extract relevant information",
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
            llm=self.llama31_8b,
        )
        
    def engagement_analysis_agent(self):
        backstory = "You evaluates content by analyzing user interactions, such as likes, \
                    shares, comments, and views. You should also consider factors such as \
                    user behavior/sentiment."
        goal = "Accurately assess the impact and effectiveness of content based on user interactions"
        return Agent(
            role="Engagement Analyst",
            goal=goal,
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
            llm=self.llama31_8b,
        )

    def relevance_analysis_agent(self):
        backstory = "You evaluates content by analyzing its alignment with {self.product_short}. \
                    You should also consider factors such as keyword density, context \
                    accuracy, and user intent. You identifies content that effectively \
                    meets audience expectations, flags irrelevant material, and \
                    provides insights to enhance content targeting."
        goal = "Ensure that content is highly pertinent and aligned with the intended topics and audience needs"
        return Agent(
            role="Relevance Analyst",
            goal=goal,
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
            llm=self.llama31_8b,
        )
        
    def content_review_supervisor_agent(self):
        backstory = "Responsible for synthesizing and evaluating the combined outputs from the \
                     Content Analysis, Engagement Analysis, and Relevance Analysis agents. You \
                     ensures all aspects of content—quality, engagement, and relevance—are \
                     harmonized and aligns with the marketing of {self.product_short}."
        goal = "Ensure content is relevant, engaging, and strategically aligned to marketing of product"
        return Agent(
            role="Content Review Supervisor",
            goal=goal,
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
            llm=self.gemma2_9b,
        )