# llm = ChatOpenAI(
#     model="crewai-llama3.1",
#     base_url="http://localhost:11434/v1",
#     openai_api_key="NA"
# )

import os
import yaml
import json
import praw
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from textwrap import dedent
from agents import *
from tasks import *
from reddit_helper import *    

class Analysis_and_ScoringCrew:
    def __init__(self):
        with open(self.cfg_file_path, 'r') as yaml_file:
            self.cfg = yaml.safe_load(yaml_file)
        self.product_long = self.cfg.get('product_long_description', '')      
        self.product_short = self.cfg.get('product_short_description', '')        

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = PostRankingAgents()
        tasks = PostRankingTasks()

        content_analysis_agent    = agents.content_analysis_agent()
        engagement_analysis_agent = agents.engagement_analysis_agent()
        relevance_analysis_agent  = agents.relevance_analysis_agent()
        content_review_agent      = agents.content_review_supervisor_agent()

        # Custom tasks include agent name and variables as input
        content_analysis_task = tasks.content_analysis_task(
            content_analysis_agent,
        )

        engagement_analysis_task = tasks.engagement_analysis_task(
            engagement_analysis_agent,
        )
        
        relevance_analysis_task = tasks.relevance_analysis_task(
            relevance_analysis_agent,
        )
        
        final_scoring_task = tasks.supervisor_review_task(
            content_review_agent,
        )        

        # Define your custom crew here
        crew = Crew(
            agents=[content_analysis_agent, engagement_analysis_agent, relevance_analysis_agent,
                    content_review_agent],
            tasks=[content_analysis_task, engagement_analysis_task, relevance_analysis_task,
                   final_scoring_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":

    print('**START**')

    reddit_posts, reddit_post_ids = fetch_reddit()
    condensed_reddit_data, unique_post_ids, unique_comment_ids = condense_data(reddit_posts, reddit_post_ids)

    analysis_scoring_crew = Analysis_and_ScoringCrew()
    scoring_result = analysis_scoring_crew.run()
    print(scoring_result)