from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langchain.tools import tool
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from typing import Literal
import os
import resend

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

class Feedback(BaseModel):
    participant_name: str = Field(description="Name of the participant")
    summary: str = Field(description="Brief summary of the overall feedback")
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the feedback like positive or negative")
    highlights: list[str]=Field(description="List of positive highlights of the program described by the participant")
    lowlights: list[str]=Field(description="List of negative highlights of the program described by the participant")
    rating: int = Field(description="Rating of the program")
    email_address: str = Field("Email address of the participant")

@tool
def send_email(email_address: str, body: str):
    """
    Tool for sending an email using resend API
    """
    resend.api_key=os.getenv("RESEND_API")
    params = resend.Emails.SendParams = {
        "from": "training@resend.dev",
        "to": [email_address],
        "subject": "Reply from Training Manager",
        "html": body
    }
    email = resend.Emails.send(params)
    return email
    

agent = create_agent(
    model=llm,
    system_prompt="You are an AI assistant who analyze the customer feedback and send an email to customer based on feedback sentiment. If sentiment is positive, you send a thank you email and if sentiment is negative you send an apology email. You use tool to send email. Use html format to draft an email.",
    response_format=Feedback,
    tools=[send_email],
)

result = agent.invoke(
    {
        "messages":[
            {"role": "user", "content": "The Java Fullstack training program did not meet expectations. Although it covered key technologies like Core Java, Spring Boot, Hibernate, and Angular, several sections were rushed—especially Spring Boot and Angular—leaving little time for hands-on practice. The lack of dedicated sessions on debugging and code optimization further reduced the program’s effectiveness. While the trainer was knowledgeable, the pace and structure limited meaningful learning and real-world application. Overall, the program requires better pacing and more practical focus. I would rate it 2 out of 5. Feedback given by Dhiraj Kumar. Email address - dhiraj2001@gmail.com"}
        ]
    }
)