from typing import List
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from blackwell.anamnesis import AnamnesisAgent
from blackwell.evaluator import EvaluatorAgent

app = FastAPI(title="Blackwell Clinical Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatHistoryRequest(BaseModel):
    thread_id: str


class ChatResponse(BaseModel):
    thread_id: str
    messages: List[dict]
    finished: bool = False


class EvaluationRequest(BaseModel):
    report: str
    thread_id: str


class EvaluationResponse(BaseModel):
    evaluation: str


def _serialize_messages(messages: List[BaseMessage]) -> List[dict]:
    serialized: List[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            continue
        serialized.append({"role": role, "content": message.content})
    return serialized


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/anamnesis", status_code=307)


@app.get("/anamnesis", response_class=HTMLResponse)
async def anamnesis_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("anamnesis.html", {"request": request})


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("evaluation.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid4())
    state = {"more_research": False, "messages": [HumanMessage(content=request.message)]}
    try:
        result = await run_in_threadpool(
            AnamnesisAgent.invoke,
            state,
            {"configurable": {"thread_id": thread_id}},
        )
        snapshot = await run_in_threadpool(
            AnamnesisAgent.get_state,
            {"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history = []
    if snapshot and getattr(snapshot, "values", None):
        history = snapshot.values.get("messages", [])
    elif result:
        history = result.get("messages", [])

    messages = _serialize_messages(history)
    finished = bool(
        messages
        and messages[-1]["role"] == "assistant"
        and messages[-1]["content"].startswith("[ANAMNESIS REPORT]")
    )

    return ChatResponse(thread_id=thread_id, messages=messages, finished=finished)


@app.post("/api/chat/history", response_model=ChatResponse)
async def get_chat_history(request: ChatHistoryRequest) -> ChatResponse:
    try:
        snapshot = await run_in_threadpool(
            AnamnesisAgent.get_state,
            {"configurable": {"thread_id": request.thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history = []
    if snapshot and getattr(snapshot, "values", None):
        history = snapshot.values.get("messages", [])

    messages = _serialize_messages(history)
    finished = bool(
        messages
        and messages[-1]["role"] == "assistant"
        and messages[-1]["content"].startswith("[ANAMNESIS REPORT]")
    )

    return ChatResponse(thread_id=request.thread_id, messages=messages, finished=finished)


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    print(f"\n=== Evaluation Request ===")
    print(f"Thread ID: {request.thread_id}")
    print(f"Report length: {len(request.report)}")
    
    initial_state = {
        "context": [],
        "anamnesis_report": HumanMessage(content=request.report),
        "query": None,
        "final_report": None,
    }
    try:
        print("Invoking Evaluator Agent...")
        result = await run_in_threadpool(
            EvaluatorAgent.invoke,
            initial_state,
            {"configurable": {"thread_id": request.thread_id}},
        )
        print(f"Evaluator Agent completed.")
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error during evaluation: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final_report = result.get("final_report", [])
    
    if not final_report:
        raise HTTPException(status_code=500, detail="Evaluator returned no report")

    message = final_report[0]
    evaluation_text = message.content if isinstance(message, BaseMessage) else str(message)
    print(f"Evaluation text length: {len(evaluation_text)}")
    print(f"=== Evaluation Complete ===\n")
    
    return EvaluationResponse(evaluation=evaluation_text)
