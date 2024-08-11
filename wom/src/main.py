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
from tasks import CustomTasks
from reddit_helper import *    

class CustomCrew:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = CustomAgents()
        tasks = CustomTasks()

        # Define your custom agents and tasks here
        custom_agent_1 = agents.agent_1_name()
        custom_agent_2 = agents.agent_2_name()

        # Custom tasks include agent name and variables as input
        custom_task_1 = tasks.task_1_name(
            custom_agent_1,
            self.var1,
            self.var2,
        )

        custom_task_2 = tasks.task_2_name(
            custom_agent_2,
        )

        # Define your custom crew here
        crew = Crew(
            agents=[custom_agent_1, custom_agent_2],
            tasks=[custom_task_1, custom_task_2],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":

    print('**START**')

    reddit_posts, reddit_post_ids = fetch_reddit()
    condensed_reddit_data, unique_post_ids, unique_comment_ids = condense_data(reddit_posts, reddit_post_ids)

    print(len(condensed_reddit_data))
    # custom_crew = CustomCrew(var1, var2)
    # result = custom_crew.run()
    # print(result)