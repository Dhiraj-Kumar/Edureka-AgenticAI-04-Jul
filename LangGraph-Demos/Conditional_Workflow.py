from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openrouter import ChatOpenRouter
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

# Pydantic Model
class ScreeningModel(BaseModel):
    candidate_name: str = Field("Name of the candidate")
    job_title: str = Field("Job title mentioned in job description")
    candidate_experience: float = Field("Working experience of the candidate as per the resume")
    experience_required: float = Field("Working experience required for the job as per the job description")
    skill_match: float = Field("Skill match score of the candidate. Value must be between 0 and 1", ge=0, le=1)
    email: str = Field("Email address of the candidate")

class ScreeningState(TypedDict):
    candidate_name: str = Field("Name of the Candidate")
    job_title: str = Field("Job title mentioned in job description")
    candidate_experience: float = Field("Working experience of the candidate as per the resume")
    experience_required: float = Field("Working experience required for the job as per the job description")
    skill_match: float = Field("Skill match score of the candidate. Value must be between 0 and 1", ge=0, le=1)
    resume_text: str = Field("Resume text passed as an input by the user")
    job_description: str= Field("Job description text passed as an input by the user")
    email: str = Field("Email address of the candidate")

structured_model = llm.with_structured_output(ScreeningModel)

def AnalyzeResumeWithJD(state: ScreeningState) -> ScreeningState:
    prompt = f"""Analyze the provided resume text and job description to extract the candidate name and total years of experience from resume, and extract the job title and required years of experience from the job description. Compare the candidate's skills with the job requirements and compute a skill_match score as a float value between 0.0 and 1.0, where 0.0 indicates no relevat skills matches the job description and 1.0 indicates strong alignment with most required skills, prioritizing skill relevance over job title.
    Resume Text:
    {state['resume_text']}
    \n
    Job Description:
    {state['job_description']}
    """
    output = structured_model.invoke(prompt)
    return {'candidate_name': output.candidate_name, "skill_match": output.skill_match, "candidate_experience": output.candidate_experience, "experience_required": output.experience_required, "job_title": output.job_title}

def CheckCriteria(state: ScreeningState) -> Literal["ShortListMail", "RejectMail"]:
    if state["skill_match"] >= 0.50 and state['candidate_experience'] >= state['experience_required']:
        return "ShortListMail"
    else:
        return "RejectMail"

def ShortListMail(state: ScreeningState) -> ScreeningState:
    prompt = f"Draft a mail to {state['candidate_name']} stating that his/her resume is shortlisted for the post of {state['job_title']}. Maintain a professional tone."
    result = llm.invoke(prompt)
    return {'email': result.content}

def RejectMail(state: ScreeningState) -> ScreeningState:
    prompt = f"Draft a mail to {state['candidate_name']} stating that his/her resume is rejected for the post of {state['job_title']}. Maintain a polite and professional tone."
    result = llm.invoke(prompt)
    return {'email': result.content}

builder = StateGraph(ScreeningState)
builder.add_node('AnalyzeResumeWithJD', AnalyzeResumeWithJD)
builder.add_node('ShortListMail', ShortListMail)
builder.add_node('RejectMail', RejectMail)

builder.add_edge(START, 'AnalyzeResumeWithJD')
builder.add_conditional_edges('AnalyzeResumeWithJD', CheckCriteria)
builder.add_edge('ShortListMail', END)
builder.add_edge('RejectMail', END)

graph = builder.compile()

image = graph.get_graph().draw_mermaid_png()
with open("conditional_graph.png", mode="wb") as f:
    f.write(image)

response = graph.invoke({'resume_text':"""
# Ankit Verma

## Professional Summary

Aspiring Software Engineer with a strong foundation in software development, object-oriented programming, and modern web technologies. Passionate about designing and developing scalable applications with clean, maintainable code. Familiar with frontend and backend development, RESTful APIs, databases, and version control. A quick learner with strong analytical and problem-solving skills, eager to contribute to innovative software development projects in a collaborative environment.

---

## Technical Skills

**Programming Languages**

* Java
* JavaScript
* SQL

**Frontend Technologies**

* HTML5
* CSS3
* JavaScript (ES6+)
* Bootstrap
* React.js

**Backend Technologies**

* Spring Boot
* REST APIs
* Node.js
* Express.js

**Database**

* MySQL
* MongoDB

**Tools & Technologies**

* Git
* GitHub
* Visual Studio Code
* IntelliJ IDEA
* Postman

**Software Engineering Concepts**

* Object-Oriented Programming (OOP)
* Data Structures & Algorithms
* MVC Architecture
* API Integration
* Exception Handling
* Debugging & Testing

---

## Technical Projects

### Employee Management System

* Developed a full-stack web application to manage employee records.
* Implemented CRUD operations using Spring Boot and MySQL.
* Designed responsive user interfaces using React.js.
* Integrated REST APIs for seamless frontend-backend communication.

### Task Management Application

* Built a task tracking application with user authentication and task categorization.
* Implemented RESTful APIs using Node.js and Express.js.
* Used MongoDB for data persistence.
* Applied Git for version control and collaborative development.

### Weather Information Dashboard

* Developed a responsive web application that retrieves real-time weather information using third-party APIs.
* Implemented API integration, asynchronous JavaScript, and responsive UI components.

---

## Internship / Training

### Full Stack Java Development Training

* Developed web applications using Java, Spring Boot, React.js, and MySQL.
* Built RESTful APIs and integrated frontend with backend services.
* Worked with Git for source code management.
* Gained hands-on experience in database design, debugging, and software development best practices.

---

## Education

**Bachelor of Arts in Business Administration**
XYZ College
2016 – 2019
   
""", 'job_description':"""
Job Title: Software Engineer – Backend

Job Summary
We are looking for a Backend Software Engineer to design, develop, and maintain scalable server-side applications. The ideal candidate will work closely with frontend developers, product managers, and DevOps teams to deliver reliable and high-performance systems.

Key Responsibilities
Design and develop backend services and APIs
Write clean, maintainable, and efficient code
Optimize applications for performance and scalability
Integrate databases, third-party services, and APIs
Participate in code reviews and system design discussions
Troubleshoot and debug production issues

Required Skills
Strong proficiency in Python or Java
Experience with backend frameworks such as Django, FastAPI, or Spring Boot
Solid understanding of REST APIs and microservices architecture
Experience with SQL databases (PostgreSQL, MySQL)
Basic knowledge of Docker and containerization
Familiarity with Git and CI/CD pipelines

Experience & Qualifications
2–5 years of professional backend development experience
Bachelor’s degree in Computer Science or a related field (or equivalent practical experience)
Experience working in an Agile/Scrum environment
Understanding of system design and scalability concepts

Nice-to-Have Skills
Experience with cloud platforms (AWS, GCP, or Azure)
Knowledge of NoSQL databases (MongoDB, Redis)
Exposure to message queues (Kafka, RabbitMQ)                          
"""})

print(response['email'])