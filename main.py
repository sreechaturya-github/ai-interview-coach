from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json
import os

app = FastAPI(title="AI Interview Prep Coach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
    temperature=0.7,
)


class JobDescRequest(BaseModel):
    job_description: str
    difficulty: str = "Medium"


class MoreQuestionsRequest(BaseModel):
    job_description: str
    category: str
    existing_questions: list[str]
    difficulty: str = "Medium"


class EvaluateRequest(BaseModel):
    job_description: str
    question: str
    answer: str


question_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical recruiter and interview coach.
Generate exactly 5 {difficulty}-level questions for EACH of the 3 categories: Technical, Behavioral, and Situational.

Difficulty guide:
- Easy: Basic concepts, simple past experiences, straightforward scenarios. Good for freshers or entry-level.
- Medium: Moderate depth, requires some real experience, involves trade-offs or decisions.
- Hard: Deep expertise, complex problem-solving, leadership, system design, or high-pressure scenarios.

Return ONLY a JSON object, no markdown, no extra text:
{{
  "Technical": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "Behavioral": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "Situational": ["question 1", "question 2", "question 3", "question 4", "question 5"]
}}
"""),
    ("human", "Job Description:\n{job_description}")
])

more_questions_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert interview coach.
Generate exactly 5 MORE {difficulty}-level {category} interview questions for the given job description.
Do NOT repeat any of the existing questions provided.

Return ONLY a JSON array of 5 strings, no markdown, no extra text:
["question 1", "question 2", "question 3", "question 4", "question 5"]
"""),
    ("human", """Job Description: {job_description}

Existing questions (do not repeat):
{existing_questions}""")
])

evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior interview coach who gives honest, constructive feedback.
Evaluate the candidate's answer and return ONLY a JSON object, no markdown, no extra text:
{{
  "score": <integer 1-10>,
  "verdict": "<Poor / Needs Work / Good / Strong / Excellent>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<improvement 1>", "<improvement 2>"],
  "sample_answer": "<A strong 3-4 sentence model answer>"
}}"""),
    ("human", """Job Description: {job_description}
Question: {question}
Answer: {answer}""")
])


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.post("/generate-questions")
async def generate_questions(req: JobDescRequest):
    chain = question_prompt | llm
    result = chain.invoke({
        "job_description": req.job_description,
        "difficulty": req.difficulty
    })
    try:
        content = result.content if isinstance(result.content, str) else str(result.content)
        questions = json.loads(content)
        return {"questions": questions}
    except Exception:
        return {"error": "Failed to parse questions.", "raw": result.content}


@app.post("/more-questions")
async def more_questions(req: MoreQuestionsRequest):
    chain = more_questions_prompt | llm
    result = chain.invoke({
        "difficulty": req.difficulty,
        "category": req.category,
        "job_description": req.job_description,
        "existing_questions": "\n".join(req.existing_questions)
    })
    try:
        content = result.content if isinstance(result.content, str) else str(result.content)
        questions = json.loads(content)
        return {"questions": questions}
    except Exception:
        return {"error": "Failed to parse questions.", "raw": result.content}


@app.post("/evaluate-answer")
async def evaluate_answer(req: EvaluateRequest):
    chain = evaluate_prompt | llm
    result = chain.invoke({
        "job_description": req.job_description,
        "question": req.question,
        "answer": req.answer,
    })
    try:
        content = result.content if isinstance(result.content, str) else str(result.content)
        feedback = json.loads(content)
        return {"feedback": feedback}
    except Exception:
        return {"error": "Failed to parse feedback.", "raw": result.content}