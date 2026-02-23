import os
import uuid
import yaml
from crewai import LLM
from pydantic import BaseModel
from typing import Optional
from crewai import Agent, Task
from firecrawl import FirecrawlApp
from pydantic import BaseModel
from pathlib import Path
from crewai import Crew, Process
from crewai_tools import DirectoryReadTool, FileReadTool 
from crewai.flow.flow import Flow, listen, or_, start, and_,router
from firecrawl import Firecrawl
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = LLM(model="gpt-4o-mini")
# from google import genai
# from google.genai import Client

# client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

# for model in client.models.list():
#     print(model.name)


import yaml

with open('crew_ai_agents/config/linkdin_agents.yaml', 'r') as f:
    agents_config = yaml.safe_load(f)

with open('crew_ai_agents/config/linkdin_tasks.yaml', 'r') as f:
    tasks_config = yaml.safe_load(f)
    
from crewai_tools import DirectoryReadTool, FileReadTool 

all_tools = [DirectoryReadTool(), FileReadTool()]

#--------------- DEFINE OUTPUT STRUCUTRE ------------------

class LinkedInPost(BaseModel):
    """Represents a LinkedIn post"""
    content: str  # The main content of the post
    media_url: str # Main image URL for the post


class ContentPlanningState(BaseModel):
    """State for the content planning flow"""
    blog_post_url: str = "https://datalemur.com/blog/statistics-interview-questions-data-science"
    draft_path: str = "crew_ai_agents/assets/"

    # Determines whether to create a Twitter or LinkedIn post 
    post_type: str = "linkedin"  
        
    # Example LinkedIn posts for reference
    path_to_example_linkedin: str = "crew_ai_agents/assets/example_linkedin.txt"


#--------------- DEFINE CREW AGENT AND TASK ------------------

draft_analyzer = Agent(config=agents_config['draft_analyzer'],
                    tools=all_tools,
                    llm=llm)

analyze_draft = Task(config=tasks_config['analyze_draft'],
                    agent=draft_analyzer
                    )

linkedin_post_planner = Agent(config=agents_config['linkedin_post_planner'],
                            tools=all_tools,
                            llm=llm)

create_linkedin_post_plan = Task(config=tasks_config['create_linkedin_post_plan'],
                                agent=linkedin_post_planner,
                                output_pydantic=LinkedInPost)




#--------------- DEFINE AGENT EXECUTION FLOW ------------------

class ContentPlanning(Flow[ContentPlanningState]):
    
    @start()
    def scrape_blog_content(self):
        print(f"# Fetching draft from: {self.state.blog_post_url}")
        
        app = FirecrawlApp(api_key = os.getenv("FIRECRAWL_API_KEY"))
        Firecrawl
        # scrape_results = app.crawl_url(self.state.blog_post_url, params = {'format':['html','markdown']})
        scrape_results = app.scrape(
            self.state.blog_post_url,
            formats=["markdown", "html"])
        #try and except blocks because calling an api that can fail
        try:
            title = scrape_results['metadata']['title']
        except Exception:
            title = str(uuid.uuid4())
            
        self.state.draft_path = f"{"crew_ai_agents/assets/"}{title}.md"
        os.makedirs(os.path.dirname(self.state.draft_path), exist_ok=True)
        # Save just the markdown if available
        markdown_text = getattr(scrape_results, "markdown", None)

        if markdown_text is None:
            # fallback to text/html or raw text if needed
            markdown_text = getattr(scrape_results, "text", "")

        with open(self.state.draft_path, 'w') as f:
            f.write(markdown_text)
            
        return self.state
    
    @router(scrape_blog_content)
    def select_platform(self):
    
        if self.state.post_type == "linkedin":
            return "linkedin"
        
    @listen("linkedin")
    def linkedin_draft(self):
        print(f"# Planning content for: {self.state.draft_path}")
        linkedin_planning_crew = Crew(agents=[linkedin_post_planner],
        tasks=[create_linkedin_post_plan], process = Process.sequential) 
        
        # Execute the LinkedIn Planning Crew
        result = linkedin_planning_crew.kickoff(inputs={
            'draft_path': self.state.draft_path, 
            'path_to_example_linkedin': self.state.path_to_example_linkedin
        })

        print(f"# Planned content for {self.state.draft_path}:")
        print(f"{result.pydantic.content}")
    
        return result



# Initialize the flow with all required fields
flow = ContentPlanning()

# Manually create StateWithId with required fields
flow.kickoff()
