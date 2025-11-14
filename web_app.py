from typing import List
from unittest import result
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
from blackwell.utils import format_references

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
        "references": [],
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
    
    if not final_report:
        raise HTTPException(status_code=500, detail="Evaluator returned no report")
    
    message = final_report[0]
    evaluation_text = message.content if isinstance(message, BaseMessage) else str(message)
    
    #evaluation_text = """***DISCLAIMER: This is an AI-generated analysis and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. All clinical decisions must be made by a qualified healthcare provider.***\n\n# **Comprehensive Clinical Analysis Report**\n\n---\n\n### **1. Introduction and Patient Presentation (Subjective)**\nThe patient presents for evaluation of a persistent and progressively worsening dermatological condition on the face. The chief complaint is the presence of "spots and bumps" which have become more constant and severe over the past four to six months. The patient describes a history of having minor, sporadic spots in their youth, but the current presentation is significantly more pronounced and unremitting. The lesions are primarily located on the forehead, nose, and chin, a distribution commonly referred to as the "T-zone," with occasional involvement of the cheeks.\n\nThe character of the lesions is described as a combination of tender red bumps and tiny black dots, particularly noticeable on the nose and chin. A significant concern for the patient is the residual marking that remains after a lesion resolves. The condition is not associated with pruritus (itching). The patient has identified stress as a potent aggravating factor, noting an almost immediate emergence of new lesions during stressful periods. Furthermore, mechanical manipulation, such as picking at the spots, is reported to exacerbate the inflammation, making them appear "angrier" and significantly delaying their healing time. Individual lesions may persist from several days to over a week. The patient describes the condition as a "constant stream" or a "never-ending rotation" of new spots, leading to a subjective severity rating of "frustrating" and "annoying to look at." Attempts at self-treatment with various over-the-counter creams and facial washes have yielded no significant improvement.\n\n### **2. Clinical Findings and History Review**\nA review of the patient\'s provided past medical history reveals a background of minor skin blemishes in youth, but no other significant diagnosed medical conditions that would predispose them to this current, more severe presentation. The patient\'s family history is not significant for similar skin conditions or other relevant medical issues.\n\nCurrently, the patient is not on any prescribed medications, supplements, or herbal remedies that could be contributing to the dermatological findings. Their current regimen is limited to the aforementioned over-the-counter topical products which have proven ineffective. The patient reports no known allergies to medications, foods, or other substances. This lack of known allergies provides a broader range of therapeutic options for consideration.\n\nThe social history is notable for an occupation as an administrative assistant, which can be stressful, corroborating the patient\'s report of stress as a primary trigger. The patient is a non-smoker, consumes alcohol lightly, and denies the use of recreational drugs. A brief review of systems was negative for any constitutional symptoms such as fever, chills, or unexplained weight changes, suggesting the condition is localized to the skin. These factors are critical in formulating a safe and effective diagnostic and treatment pathway.\n\n### **3. Diagnostic Analysis and Differential**\nGiven the full symptomatic picture, a differential diagnosis was formulated. The patient\'s presentation of tender red bumps (inflammatory papules), tiny black dots (open comedones), and their specific distribution in the sebaceous gland-rich T-zone strongly suggests a primary diagnosis rooted in a disorder of the pilosebaceous unit.\n\n**Probable Cause: Acne Vulgaris**\n\n*   **Detailed Rationale:** The evidence supporting Acne Vulgaris as the primary diagnosis is substantial and aligns perfectly with the classic clinical presentation. The patient\'s description of "red bumps" corresponds to inflammatory papules, and the "tiny black dots" are pathognomonic for open comedones (blackheads), a hallmark feature of this condition. The localization of these lesions to the forehead, nose, and chin is the most common distribution for acne due to the high density of sebaceous follicles in these areas. The progressive worsening over several months and the chronic, relapsing nature described as a "constant stream" are characteristic of the natural history of moderate acne.\n\n    Furthermore, the patient\'s report that stress is a significant exacerbating factor is well-documented in clinical literature, as cortisol and other stress hormones can increase sebum production and inflammation. The fact that various over-the-counter treatments have failed is also a key diagnostic clue, suggesting the condition has surpassed the threshold of mild acne and requires prescription-strength intervention. The residual marks noted by the patient are consistent with post-inflammatory hyperpigmentation, a common sequela of inflammatory acne lesions. The sum of these findings—comedones, inflammatory papules, T-zone distribution, chronicity, and response to known triggers—creates a compelling clinical picture for Acne Vulgaris.\n\n**Considered Differential Diagnoses:**\n\n*   **Rosacea:** This was considered as a potential alternative, particularly the papulopustular subtype, due to the patient\'s report of facial redness and "pimple-like bumps." Rosacea is a chronic inflammatory condition that can mimic acne. However, a key differentiating feature makes this diagnosis less likely: the presence of comedones. The patient explicitly describes "tiny black dots," which are a defining feature of acne vulgaris and are characteristically absent in rosacea. While both conditions can be exacerbated by triggers, the overall presentation, especially the comedonal component, points much more strongly toward acne.\n\n*   **Hidradenitis Suppurativa (Acne Inversa):** This was also evaluated but is considered a very low-probability diagnosis. While its alternative name, "acne inversa," suggests a relationship, its clinical presentation is markedly different. Hidradenitis suppurativa is a chronic inflammatory disease of the hair follicles that typically occurs in intertriginous areas of the body, such as the axillae (armpits), groin, and inframammary regions. It presents with deep, painful, inflammatory nodules, abscesses, and sinus tracts. The patient\'s lesions are superficial (bumps and dots), located exclusively on the face, and do not match the morphology or distribution of hidradenitis suppurativa.\n\n### **4. Recommended Management Plan (Plan)**\nBased on the probable diagnosis of Acne Vulgaris, the following comprehensive management plan is recommended, focusing on treating existing lesions, preventing the formation of new ones, and addressing the patient\'s specific contributing factors.\n\n**Patient-Specific Considerations:**\nIt is noted that the patient\'s condition has been unresponsive to over-the-counter therapies, indicating a need to escalate to prescription-strength agents. The patient has no known medication allergies, which allows for a wide range of therapeutic choices. The significant impact of stress from their occupation and the detrimental habit of picking at lesions are critical behavioral targets that must be addressed in parallel with pharmacological therapy for a successful outcome.\n\n**1. Pharmacological Therapeutic Strategy:**\nA multi-pronged pharmacological approach is warranted to target the different pathogenic factors of acne: follicular hyperkeratinization, excess sebum, bacterial proliferation, and inflammation.\n\n*   **First-Line Topical Therapy:** A topical retinoid (e.g., tretinoin, adapalene) is strongly recommended. The rationale for this choice is its high efficacy in treating both comedonal acne (the "black dots") and inflammatory lesions. Retinoids work by normalizing follicular keratinization, which helps to unclog pores and prevent the formation of new microcomedones. This should be applied thinly to the entire affected area in the evening.\n*   **Adjunctive Topical Therapy:** To address the inflammatory "red bumps" and bacterial component, a combination product containing benzoyl peroxide and a topical antibiotic (e.g., clindamycin) is recommended for morning application. Benzoyl peroxide is a potent antimicrobial agent that also helps prevent the development of bacterial resistance to the antibiotic.\n*   **Consideration for Systemic Therapy:** If the response to aggressive topical therapy is suboptimal after 2-3 months, or if the inflammation is more severe than initially appreciated, the addition of an oral antibiotic (e.g., doxycycline, minocycline) for a defined course would be the next logical step to provide systemic anti-inflammatory and antibacterial effects.\n\n**2. Non-Pharmacological and Lifestyle Interventions:**\nPharmacological intervention is only one part of a successful management strategy. The following adjunctive measures are paramount to achieving and maintaining clearance.\n\n*   **Gentle Skincare Protocol:** The patient should cease using multiple, potentially harsh OTC products and adopt a simple, gentle skincare routine. This includes washing the face twice daily with a mild, non-comedogenic cleanser and using an oil-free, non-comedogenic moisturizer to counteract the potential drying effects of the topical medications.\n*   **Strict Avoidance of Manipulation:** The patient explicitly noted that picking makes lesions "angrier" and prolongs healing. This behavior can introduce secondary bacterial infection, increase inflammation, and significantly raise the risk of permanent scarring and post-inflammatory hyperpigmentation. Counseling on the importance of strict avoidance is a critical behavioral component of the management plan.\n*   **Stress Management Techniques:** Given that stress is a powerful trigger for this patient, identifying and implementing stress-reduction strategies is essential for long-term control. This could include mindfulness exercises, regular physical activity, ensuring adequate sleep, or developing coping mechanisms specifically for high-stress periods at work.\n*   **Sun Protection:** Topical retinoids can increase the skin\'s sensitivity to the sun. Daily use of a broad-spectrum, non-comedogenic sunscreen with an SPF of 30 or higher is crucial to protect the skin and prevent worsening of post-inflammatory marks.\n\n**3. Follow-up and Monitoring:**\nThe patient should be scheduled for a follow-up appointment in 8-12 weeks to assess their response to the initial treatment plan, monitor for any potential side effects from the medications, and make adjustments as necessary. It is vital to educate the patient that visible improvement with acne treatment is gradual and requires consistent adherence to the regimen.\n\n### **5. Concluding Summary**\nIn summary, this patient presents with a classic clinical picture of moderate, mixed comedonal and inflammatory Acne Vulgaris, localized to the facial T-zone. The diagnosis is strongly supported by the morphology of the lesions, their distribution, the chronic course, and the failure to respond to over-the-counter remedies. The recommended management plan is a comprehensive, multi-modal approach that combines evidence-based prescription topical therapies to target the underlying pathophysiology of acne with essential non-pharmacological strategies. These include a gentle skincare routine, strict avoidance of lesion manipulation, and proactive stress management. Consistent follow-up with a healthcare provider is necessary to monitor progress and optimize the therapeutic regimen for long-term control and prevention of scarring."""
    #references_text = """### References\n\n**Knowledge Base References:**\n- MedlinePlus: Acne (https://medlineplus.gov/acne.html)\n- MedlinePlus: Rosacea (https://medlineplus.gov/rosacea.html)\n- MedlinePlus: Hidradenitis Suppurativa (https://medlineplus.gov/hidraditissuppurativa.html)\n- Mayo Clinic (General Information)\n- MedlinePlus (Acne, Over-the-Counter Medicines)\n- American Academy of Dermatology (AAD) (Acne: Tips for Managing, Acne Scars)\n- National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) (Acne)\n- FamilyDoctor.org (Acne)\n- Nemours Foundation / KidsHealth.org (Acne)\n\n**PubMed Articles:**\n- The Burden of Acne Vulgaris on Health-Related Quality of Life and Psychosocial Well-Being Domains: A Systematic Review. (PMID: 41134527)\n- Use of Dermocosmetics in Acne Management: A Middle East-North Africa Consensus. (PMID: 40932054)\n- PRACT-India: Practical Recommendations on Acne Care and Medical Treatment in India-A Modified Delphi Consensus. (PMID: 40868038)\n- Effectiveness of a Machine Learning-Enabled Skincare Recommendation for Mild-to-Moderate Acne Vulgaris: 8-Week Evaluator-Blinded Randomized Controlled Trial. (PMID: 40669065)\n- Competencies and clinical guidelines for managing acne with isotretinoin in general practice: a scoping review. (PMID: 40562445)\n- An update on the pharmacological management of acne vulgaris: the state of the art. (PMID: 39420562)\n- Guidance for the pharmacological management of acne vulgaris. (PMID: 34686076)\n- Pharmacological Management and Potentially Inappropriate Prescriptions for Patients with Acne. (PMID: 38912194)\n- Topical clindamycin for acne vulgaris: analysis of gastrointestinal events. (PMID: 38568005)\n- Isotretinoin in the management of acne vulgaris: practical prescribing. (PMID: 32860434)\n"""
    # Format references for inclusion in the final report
    references_text = format_references(result["references"])
    evaluation_text += f"\n\n{references_text}"
    #print(evaluation_text)
    print(f"Evaluation text length: {len(evaluation_text)}")
    print(f"=== Evaluation Complete ===\n")
    
    return EvaluationResponse(evaluation=evaluation_text)
