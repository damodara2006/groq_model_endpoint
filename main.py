"""
GenAI Model Endpoint — LangChain + Groq
Exposes a /summarize endpoint that the AI Governance Framework calls
to get live generated_answer values for Ragas faithfulness checks.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(
    title="GenAI Model Endpoint (doc_summarizer)",
    description="LangChain + Groq powered document summarizer for AI governance testing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq LLM ──────────────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    return ChatGroq(
        api_key=api_key,
        model_name=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "512")),
    )


# ── Request / Response models ─────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    input_text: str
    system_prompt: str = (
        "You are a precise document summarizer. "
        "Summarize the given text accurately and concisely. "
        "Do not add information not present in the source text."
    )
    question: str = ""   # optional — governs what aspect to focus on


class SummarizeResponse(BaseModel):
    generated_answer: str
    model_used: str
    input_length: int
    output_length: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    """
    Main endpoint called by the AI Governance Framework.
    POST body: { input_text, system_prompt, question }
    Returns:   { generated_answer, model_used, input_length, output_length }
    """
    if not req.input_text.strip():
        raise HTTPException(status_code=400, detail="input_text is required")

    try:
        llm = get_llm()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Build the prompt — if question provided, guide the answer toward it
    user_content = req.input_text
    if req.question:
        user_content = f"Question: {req.question}\n\nDocument:\n{req.input_text}"

    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    return SummarizeResponse(
        generated_answer=answer,
        model_used=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        input_length=len(req.input_text),
        output_length=len(answer),
    )
