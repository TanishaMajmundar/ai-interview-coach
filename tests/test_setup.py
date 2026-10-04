import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from app.models import CandidateProfile, Evaluation

load_dotenv()

# Test 1: can we call the LLM?
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
reply = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=500,
)
print("LLM says:", reply.choices[0].message.content)

# Test 2: do the models accept valid data?
profile = CandidateProfile(skills=["Python"], experience=["Intern"], gaps_vs_jd=["Docker"])
evaluation = Evaluation(relevance=4, depth=3, clarity=4, structure=3,
                        strengths=["Clear"], weaknesses=["Shallow"], improved_answer="...")
print("Valid:", profile, evaluation)

# Test 3: do the models reject invalid data?
try:
    Evaluation(relevance=9, depth=3, clarity=4, structure=3,
               strengths=[], weaknesses=[], improved_answer="")
except ValidationError:
    print("Invalid score correctly rejected")