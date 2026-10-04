from typing import List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    name: Optional[str] = None
    skills: List[str]
    experience: List[str]
    gaps_vs_jd: List[str]  # skills the JD wants that the resume lacks


class Question(BaseModel):
    text: str
    mode: Literal["hr", "technical", "coding"]
    difficulty: int = Field(ge=1, le=5)


class Evaluation(BaseModel):
    relevance: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    structure: int = Field(ge=1, le=5)
    strengths: List[str]
    weaknesses: List[str]
    improved_answer: str


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    interview_type: str
    average_score: float
    questions: List[Question]
    evaluations: List[Evaluation]