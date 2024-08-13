import os
import yaml
import json
import praw
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from textwrap import dedent
from reddit_helper import *    

from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

from pydantic import BaseModel

api_file_path = 'api_keys.json'
with open(api_file_path, 'r') as file:
    api_keys = json.load(file)        
openai_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", 
                                temperature=0.7,
                                openai_api_key=api_keys['openai'])
openai_gpt4 = ChatOpenAI(model_name="gpt-4", 
                                temperature=0.7,
                                openai_api_key=api_keys['openai'])        
llama31_8b = Ollama(model='llama3.1:8b',)
gemma2_9b = Ollama(model='gemma2:9b',)

cfg_file_path = 'casaai_config.yaml'
with open(cfg_file_path, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)
product_long = cfg.get('product_long_description', '')      
product_short = cfg.get('product_short_description', '')

backstory = "You are a content analyst with expertise in analyzing web content and \
                extracting relevant information. You are responsible for ensuring that \
                content is relevant, high-quality, and aligned with the marketing of \
            {product_short}. "
content_analysis_agent = Agent(
                            role="Content Analyst",
                            goal="Analyze web content and extract relevant information",
                            backstory=backstory,
                            allow_delegation=False,
                            verbose=True,
                            llm=llama31_8b,
                            )

backstory = "You evaluates content by analyzing user interactions, such as likes, \
            shares, comments, and views. You should also consider factors such as \
            user behavior/sentiment."
goal = "Accurately assess the impact and effectiveness of content based on user interactions"
engagement_analysis_agent = Agent(
                            role="Engagement Analyst",
                            goal=goal,
                            backstory=backstory,
                            allow_delegation=False,
                            verbose=True,
                            llm=llama31_8b,
                            )

backstory = "You evaluates content by analyzing its alignment with {product_short}. \
            You should also consider factors such as keyword density, context \
            accuracy, and user intent. You identifies content that effectively \
            meets audience expectations, flags irrelevant material, and \
            provides insights to enhance content targeting."
goal = "Ensure that content is highly pertinent and aligned with the intended topics and audience needs"
relevance_analysis_agent = Agent(
                            role="Relevance Analyst",
                            goal=goal,
                            backstory=backstory,
                            allow_delegation=False,
                            verbose=True,
                            llm=llama31_8b,
                        )

backstory = "Responsible for synthesizing and evaluating the combined outputs from the \
                Content Analysis, Engagement Analysis, and Relevance Analysis agents. You \
                ensures all aspects of content—quality, engagement, and relevance—are \
                harmonized and aligns with the marketing of {product_short}."
goal = "Ensure content is relevant, engaging, and strategically aligned to marketing of product"
content_review_agent = Agent(
                            role="Content Review Supervisor",
                            goal=goal,
                            backstory=backstory,
                            allow_delegation=False,
                            verbose=True,
                            llm=gemma2_9b,
                        )

class ScoreOutput(BaseModel):
    comment_id: str
    score: float
    justification: str
    
tip_text = "If you do your BEST WORK, I'll give you a $10,000 commission!"

descr = "Analyze posts and associated comments from {input_data} to determine their relevance based on \
identified keywords and phrases w.r.to the marketing of {product_long}. Then provide a score \
of 10 for each post and comment and provide a justification for each score. {tip_text}"

expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                explaining the rationale behind the score for each post and its associated comments." 
        
content_analysis_task = Task(
                            description=descr,
                            expected_output=expected_out,
                            agent=content_analysis_agent,
                            output_json=ScoreOutput,
                            )

descr = "Evaluating the level of user interaction with the provided content from {input_data}. This includes\
analyzing metrics such as likes, shares, comments, and views to calculate an overall engagement score. \
{tip_text}"       

expected_out = "JSON with comment_id, engagement score, and a brief justification (less than 15 words) \
                explaining the rationale behind the score for each post and its associated comments." 

engagement_analysis_task =  Task(
                                description=descr,
                                expected_output=expected_out,
                                agent=engagement_analysis_agent,
                                output_json=ScoreOutput,
                                )

descr = "Assess how well the content in {input_data} aligns with {product_long}. The goal is to assign a \
relevance score that reflects the content’s pertinence to its intended audience and its  \
alignment with the product that is marketed. {tip_text}"        

expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                explaining the rationale behind the score for each post and its associated comments." 
        
relevance_analysis_task = Task(
                                description=descr,
                                expected_output=expected_out,
                                agent=relevance_analysis_agent,
                                output_json=ScoreOutput,
                                )

descr = "You will get outputs from Content Analysis, Engagement Analysis, and Relevance Analysis agents. \
You will also get content from {input_data}. You will review the outputs from these agents and the content \
and provide a final score for each post and comment based on the relevance to the marketing of \
{product_long}. {tip_text}"        
       
expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 30 words) \
                explaining the rationale behind the score for each post and its associated comments." 
        
final_scoring_task = Task(
            description=descr,
            expected_output=expected_out,
            agent=content_review_agent,
            output_json=ScoreOutput,
        )   

response_creation_crew = Crew(
    agents=[content_analysis_agent, engagement_analysis_agent, relevance_analysis_agent,
            content_review_agent],
    tasks=[content_analysis_task, engagement_analysis_task, relevance_analysis_task,
            final_scoring_task],
    verbose=True,
)

reddit_posts, reddit_post_ids = fetch_reddit_test()
condensed_reddit_data, unique_post_ids, unique_comment_ids = condense_data(reddit_posts, reddit_post_ids)

input_dict = {"input_data": condensed_reddit_data}

scoring_result = response_creation_crew.kickoff(inputs=input_dict)


        
