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
    state = {"messages": [HumanMessage(content=request.message)]}
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
        "research_report": None,
        "t_run": 0,
        "query": None,
        "hypothesis_report": None,
        "treatment_report": None,
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
    #testfinal_report = """***DISCLAIMER: This is an AI-generated analysis and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. All clinical decisions must be made by a qualified healthcare provider.***\n\n# **Comprehensive Clinical Analysis Report**\n\n---\n\n### **1. Introduction and Patient Presentation (Subjective)**\n\nThe patient presents for evaluation of a new-onset dermatological complaint localized to the penis and groin. The chief complaint is characterized by redness and significant irritation. The patient provides a clear and compelling history indicating that the symptoms are directly and exclusively precipitated by mechanical friction, specifically during sexual activity or masturbation.\n\nThe history of the present illness reveals a recurrent pattern. The patient, who is circumcised, notes the appearance of erythema (redness) and, in more severe instances, the development of line-shaped fissures, which he describes as resembling small cuts. Critically, these fissures are superficial and do not bleed. The associated symptoms include a burning sensation and pain, which the patient rates as a maximum of 4 out of 10 on a standard pain scale, accompanied by occasional pruritus (itching). The clinical course is highly predictable: symptoms manifest and worsen following frictional events and subsequently improve and resolve completely when friction is avoided, allowing the skin to heal. This is the first time the patient has experienced this specific condition.\n\n### **2. Clinical Findings and History Review**\n\nA review of the patient\'s provided past medical history reveals no previously diagnosed medical conditions that would typically predispose an individual to such dermatological issues; specifically, there is no history of diabetes mellitus, eczema (atopic dermatitis), psoriasis, or known autoimmune disorders. The patient is not currently taking any medications, including over-the-counter products, prescription drugs, or herbal supplements. Furthermore, the patient reports no known allergies to medications, latex, soaps, or detergents. This lack of confounding factors is clinically significant as it helps to narrow the diagnostic possibilities away from allergic contact dermatitis or drug-induced reactions.\n\nThe patient\'s social history is notable for not using condoms with his current partner, which is an important detail when considering potential latex allergies or sexually transmitted infections in the differential diagnosis. A brief review of systems was negative for any constitutional symptoms such as fever, chills, or unexplained weight changes, and the patient denies any alterations in urination habits, making a systemic infectious process or a primary urological issue less likely. These factors are critical in formulating a safe and effective diagnostic and treatment pathway.\n\n### **3. Diagnostic Analysis and Differential**\n\nGiven the full symptomatic picture, a differential diagnosis was formulated. The patient\'s presentation of localized redness, non-bleeding fissures, and a clear, reproducible link between mechanical friction and symptom onset strongly suggests a primary dermatological condition of a traumatic or irritant nature, rather than an infectious or systemic inflammatory process.\n\n**Probable Cause: Frictional Dermatitis (Traumatic Dermatitis)**\n\n*   **Detailed Rationale:** The evidence supporting Frictional Dermatitis as the primary diagnosis is substantial and compelling. This condition is a form of mechanical injury where the skin\'s structural integrity is compromised by repetitive rubbing, stretching, or shearing forces. The patient\'s history aligns perfectly with this pathophysiology. The direct cause-and-effect relationship—symptoms appearing only after sexual activity and resolving completely with rest—is the hallmark of this diagnosis. The line-shaped fissures are characteristic of skin that has been stretched beyond its elastic limit, resulting in superficial tears in the epidermis. The fact that these fissures do not bleed indicates the injury is confined to the superficial layers of the skin, consistent with trauma rather than a deeper ulcerative process. The complete resolution upon cessation of the trigger is a powerful indicator that this is not a chronic inflammatory condition (like psoriasis) or a persistent infection, but rather a direct response to an external physical force.\n\n**Considered Differential Diagnoses:**\n\n*   **Irritant Contact Dermatitis (ICD):** This was considered as a potential alternative diagnosis. ICD is a non-allergic inflammatory skin reaction caused by exposure to a substance that directly damages the skin. While friction can certainly exacerbate ICD by driving an irritant deeper into the stratum corneum, this diagnosis is less likely to be the primary cause here. The patient explicitly denies any new exposures to potential irritants such as soaps, detergents, lubricants, or spermicides. The clinical picture is dominated by the mechanical trigger, making a purely chemical etiology improbable. If a new, unmentioned product were involved, one would expect the reaction to persist as long as the product is in use, not just immediately following friction.\n\n*   **Recurrent Herpes Simplex Virus (HSV) Infection:** This was also evaluated, as any recurrent genital lesion warrants consideration of an infectious, sexually transmitted cause. However, this diagnosis was ultimately ranked low because the patient\'s clinical description is highly inconsistent with a typical herpetic outbreak. Genital herpes classically presents with grouped vesicles (small blisters) that evolve into painful, well-demarcated ulcers before crusting over and healing. The patient describes linear "cuts" that do not bleed, which is morphologically distinct from herpetic ulcers. Furthermore, the strict and immediate correlation with mechanical trauma as the sole trigger is atypical for HSV, which recurs due to viral reactivation, often triggered by systemic stressors, not localized friction alone.\n\n**Diagnostic Uncertainty and Further Steps:**\nWhile the clinical history is strongly indicative of Frictional Dermatitis, should the lesions change in character or fail to resolve as expected with the recommended interventions, further clarification could be pursued. To definitively rule out the low-probability differential of HSV, a viral culture or a polymerase chain reaction (PCR) swab could be taken from a fresh lesion during an active flare-up.\n\n### **4. Recommended Management Plan (Plan)**\n\nBased on the probable diagnosis of Frictional Dermatitis, the following comprehensive management plan is recommended, focusing on both acute symptom relief and, more importantly, long-term prevention by addressing the root mechanical cause.\n\n**Patient-Specific Considerations:**\nIt is noted that the patient has no history of underlying medical conditions and reports no known allergies. These factors allow for a focused and simplified management plan that prioritizes the elimination of the mechanical trigger and the restoration of the skin\'s natural barrier function, without the need to account for medication interactions or allergic sensitivities.\n\n**1. Abortive (Acute) Therapeutic Strategy:**\nTo manage acute attacks and promote rapid healing, the following steps are essential.\n\n*   **Rest and Avoidance of Friction:** This is the single most important therapeutic intervention. When redness or fissures are present, the patient must temporarily cease or significantly modify sexual and masturbatory activities to allow the epidermal barrier to fully heal. Continued trauma will prevent resolution and may lead to secondary complications.\n*   **Application of Topical Emollients:** To support the healing process, it is recommended to apply a bland, fragrance-free, and hypoallergenic emollient or moisturizer to the affected area multiple times per day. The rationale for this is to hydrate the stratum corneum, reduce transepidermal water loss, and provide a soothing, protective environment that facilitates the repair of the skin barrier.\n\n**2. Non-Pharmacological and Lifestyle Interventions:**\nPharmacological intervention is not indicated at this stage. The cornerstone of successful long-term management is prevention through lifestyle and behavioral modification.\n\n*   **Ensure Adequate Lubrication:** This is the primary preventative measure. The patient must utilize a generous amount of high-quality lubricant prior to and during all sexual and masturbatory activities. Water-based lubricants are generally well-tolerated and effective. The clinical principle is simple: lubrication drastically reduces the coefficient of friction, thereby preventing the shearing and stretching forces that cause the skin trauma.\n*   **Consider Barrier Creams:** For added protection, the use of a barrier cream containing ingredients such as dimethicone or zinc oxide can be beneficial. These products form a durable, semi-occlusive film on the skin that acts as a physical shield against friction.\n*   **Gentle Hygiene and Appropriate Clothing:** The patient should avoid harsh soaps or cleansers in the genital area, as these can strip the skin of its natural protective oils and worsen irritation. Using a gentle, soap-free cleanser is advised. Additionally, wearing loose-fitting, breathable underwear made of natural fibers like cotton can help reduce ambient moisture and incidental friction throughout the day.\n\n**3. Preventative (Prophylactic) Strategy:**\nThe long-term strategy is entirely focused on preventing recurrence by managing the mechanical forces involved.\n\n*   **Conscious Modification of Practices:** The patient should become mindful of the specific activities or techniques that lead to trauma. This involves a conscious effort to ensure lubrication is always sufficient and to potentially moderate the intensity or duration of friction to remain within the skin\'s tolerance.\n*   **Follow-Up and Specialist Referral:** If the condition persists, worsens, or changes in character despite strict adherence to these conservative management strategies, a follow-up evaluation is crucial. Referral to a dermatologist or urologist would be warranted to confirm the diagnosis and rule out any other underlying dermatological or urological conditions.\n\n### **5. Concluding Summary**\n\nIn summary, the patient presents with a clear history and clinical description highly characteristic of Frictional Dermatitis of the penis and groin. The diagnosis is strongly supported by the direct and reproducible causal link between mechanical friction and the onset of symptoms, as well as the complete resolution of symptoms upon cessation of the trauma. The recommended management plan is non-pharmacological and is centered on the core principles of eliminating the offending trigger and supporting the skin\'s natural barrier function. The primary interventions include temporary rest during acute flares, consistent and generous use of lubrication during activity for prevention, and gentle skin care practices. The prognosis for this condition is excellent, with a high likelihood of complete resolution and prevention of recurrence provided the patient adheres to the recommended preventative strategies."""
    
    if not final_report:
        raise HTTPException(status_code=500, detail="Evaluator returned no report")

    message = final_report[0]
    evaluation_text = message.content if isinstance(message, BaseMessage) else str(message)
    print(f"Evaluation text length: {len(evaluation_text)}")
    print(f"=== Evaluation Complete ===\n")
    
    return EvaluationResponse(evaluation=evaluation_text)
