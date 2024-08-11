from crewai import Task
from textwrap import dedent
import json
import yaml
from pydantic import BaseModel

class ScoreOutput(BaseModel):
    comment_id: str
    score: float
    justification: str

class CustomTasks:
    def __init__(self):
        with open(self.cfg_file_path, 'r') as yaml_file:
            self.cfg = yaml.safe_load(yaml_file)
        self.product_long  = self.cfg.get('product_long_description', '')      
        self.product_short = self.cfg.get('product_short_description', '')
        self.input_format_descr = "Input format is a python list of dictionaries. There will be \
                                   a parent post and its associated comments. Parent post has \
                                   'parent_id' = None. Comments have 'parent_id' = 'comment_id' of \
                                   parent post"   
        self.tip_text = "If you do your BEST WORK, I'll give you a $10,000 commission!" 

    def content_analysis_task(self, agent):
        
        descr = "Analyze posts and associated comments from {input} to determine their relevance based on \
        identified keywords and phrases w.r.to the marketing of {self.product_long}. Then provide a score \
        of 10 for each post and comment and provide a justification for each score. {self.tip_text}"
        
        expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                        explaining the rationale behind the score for each post and its associated comments.", 
        
        return Task(
            description=descr,
            expected_output=expected_out,
            agent=agent,
            output_json=ScoreOutput,
        )

    def engagement_analysis_task(self, agent):
        
        descr = "Analyze posts and associated comments from {input} to determine their relevance based on \
        identified keywords and phrases w.r.to the marketing of {self.product_long}. Then provide a score \
        of 10 for each post and comment and provide a justification for each score. {self.tip_text}"
        
        expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                        explaining the rationale behind the score for each post and its associated comments.", 
        
        return Task(
            description=descr,
            expected_output=expected_out,
            agent=agent,
            output_json=ScoreOutput,
        )
    
    def relevance_analysis_task(self, agent):
        
        descr = "Analyze posts and associated comments from {input} to determine their relevance based on \
        identified keywords and phrases w.r.to the marketing of {self.product_long}. Then provide a score \
        of 10 for each post and comment and provide a justification for each score. {self.tip_text}"
        
        expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                        explaining the rationale behind the score for each post and its associated comments.", 
        
        return Task(
            description=descr,
            expected_output=expected_out,
            agent=agent,
            output_json=ScoreOutput,
        )   
        
    def supervisor_review_task(self, agent):
        
        descr = "Analyze posts and associated comments from {input} to determine their relevance based on \
        identified keywords and phrases w.r.to the marketing of {self.product_long}. Then provide a score \
        of 10 for each post and comment and provide a justification for each score. {self.tip_text}"
        
        expected_out = "JSON with comment_id, relevance score, and a brief justification (less than 15 words) \
                        explaining the rationale behind the score for each post and its associated comments.", 
        
        return Task(
            description=descr,
            expected_output=expected_out,
            agent=agent,
            output_json=ScoreOutput,
        ) 