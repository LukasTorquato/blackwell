from typing import List
from unittest import result
from uuid import uuid4
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
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

# Ensure tmp directory exists
TMP_DIR = Path("./tmp")
TMP_DIR.mkdir(exist_ok=True)
global state
state = {"messages": [], "documents_report": None, "final_report": None, "function": "chat"}
class ChatRequest(BaseModel):
    message: str = ""  # Make message optional with empty string default
    thread_id: str | None = None


class ChatHistoryRequest(BaseModel):
    thread_id: str


class ChatResponse(BaseModel):
    thread_id: str
    messages: List[dict]
    finished: bool = False
    final_report: str | None = None


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


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), thread_id: str = Form(...)):
    """Upload files to the tmp directory for document analysis"""
    global state
    try:
        # Clear existing files in tmp directory
        for file_path in TMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        uploaded_filenames = []
        for file in files:
            # Save file to tmp directory
            file_path = TMP_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)

        # Update state to trigger document analysis
        state["function"] = "document_analysis"
        result = await run_in_threadpool(
                AnamnesisAgent.invoke,
                state,
                {"configurable": {"thread_id": thread_id}},
            )
        del result["messages"][-2]
        state = result
        # Serialize messages for response
        messages = _serialize_messages(state["messages"])
        
        return {
            "status": "success", 
            "files": uploaded_filenames,
            "messages": messages,
            "thread_id": thread_id
        }
    except Exception as exc:
        print(f"Error during file upload: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reset")
async def reset_session():
    """Reset the global state for a new anamnesis session"""
    global state
    try:
        # Reset the global state
        state = {"messages": [], "documents_report": None, "final_report": None, "function": "chat"}
        
        # Clear any uploaded files in tmp directory
        for file_path in TMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        return {"status": "success", "message": "Session reset successfully"}
    except Exception as exc:
        print(f"Error during session reset: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid4())
    if request.message != "":
        state["messages"].append(HumanMessage(content=request.message))
        state["function"] = "chat"
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
            if "quota" in str(exc).lower():
                raise HTTPException(status_code=503, detail="Quota exceeded, please try again later.") from exc
            print(f"Error during chat: {str(exc)}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        history = []
        if snapshot and getattr(snapshot, "values", None):
            history = snapshot.values.get("messages", [])
        elif result:
            history = result.get("messages", [])
    messages = _serialize_messages(history)
    # Check if we have a final_report (anamnesis + documents analysis complete)
    final_report = result.get("final_report") if result else None
    
    return ChatResponse(
        thread_id=thread_id, 
        messages=messages, 
        finished=bool(final_report),
        final_report=final_report
    )


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
    finished = bool(snapshot.values.get("final_report"))

    return ChatResponse(thread_id=request.thread_id, messages=messages, finished=finished)


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    print(f"\n=== Evaluation Request ===")
    print(f"Thread ID: {request.thread_id}")
    print(f"Report length: {len(request.report)}")
    
    initial_state = {
        "references": [],
        "anamnesis_report": HumanMessage(content=request.report),
        "t_run": 0,
        "query": None,
        "reports": {},
        "final_report": None,
    }
    """
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
    
    evaluation_text = final_report.content if isinstance(final_report, BaseMessage) else str(final_report)
    """
    evaluation_text = """"***DISCLAIMER: This is an AI-generated analysis and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. All clinical decisions must be made by a qualified healthcare provider.***\n\n# **Comprehensive Clinical Analysis Report**\n\n---\n\n### **1. Introduction and Patient Presentation (Subjective)**\nThe patient is a 58-year-old female who presents for evaluation with a chief complaint of overwhelming fatigue and the appearance of unusual, spontaneous bruising. The onset of the fatigue was gradual, beginning several weeks ago, but has significantly worsened over the past two weeks to a point where it is profoundly impacting her quality of life. She describes this fatigue as a pervasive feeling of being \"run down\" and \"wiped out,\" assigning it a severity of 7-8 on a 10-point scale. Even simple daily tasks now require a substantial effort.\n\nConcurrently, the patient has noted the appearance of scattered bruising on her lower legs. These ecchymoses are notable because they appear without a clear history of significant trauma, sometimes resulting from what she perceives as only minor bumps. In addition to these primary complaints, she reports a sensation of heaviness and noticeable swelling in her lower legs, concentrated in the calves and ankles. She also observes that the veins on her calves have become more prominent and appear to be bulging. The leg heaviness is a constant symptom but is significantly aggravated by prolonged periods of being on her feet, characteristically worsening towards the end of the day. The only alleviating factor she has identified is the elevation of her legs, which provides some measure of relief from the sensation of heaviness. The patient has not experienced any associated systemic symptoms such as fever or chills.\n\n### **2. Clinical Findings and History Review**\nA review of the patient's provided past medical history reveals no specific diagnosed conditions that would directly account for her current constellation of symptoms, although she self-reports being overweight. Her current medication regimen is minimal, consisting of a daily multivitamin and the occasional use of ibuprofen for headaches. The patient reports no known drug allergies.\n\nThe patient's family history is significant for a father who experienced \"heart trouble\" later in life, while her mother was generally healthy. There is no known family history of specific circulation disorders, venous disease, or hematological/bleeding disorders. Socially, the patient is a non-smoker and consumes alcohol only occasionally (approximately one glass of wine every few weeks). Her occupation as a librarian is mostly sedentary but does involve some periods of walking and standing. These factors\u2014specifically her age, female sex, overweight status, and periods of prolonged sitting or standing associated with her occupation\u2014are critical in formulating a safe and effective diagnostic and treatment pathway, as they represent known risk factors for venous disease.\n\n### **3. Diagnostic Analysis and Differential**\nGiven the full symptomatic picture, a differential diagnosis was formulated. The patient's presentation of leg heaviness, lower extremity swelling, and prominent veins strongly suggests a primary vascular component, specifically venous pathology. However, the concurrent symptoms of overwhelming fatigue and spontaneous bruising demand a broader investigation to include potential systemic or hematological disorders.\n\n**Probable Cause: Chronic Venous Insufficiency (CVI) with suspected underlying coagulopathy.**\n\n*   **Detailed Rationale:** The evidence supporting Chronic Venous Insufficiency as a core component of the diagnosis is substantial. The patient's reported symptoms are classic hallmarks of this condition. The sensation of heaviness and aching in the lower legs, the visible swelling (edema) around the ankles and calves, and the observation of prominent or bulging veins (varicosities) are pathognomonic for venous hypertension resulting from incompetent venous valves. The chronicity of these symptoms and the characteristic pattern of exacerbation\u2014worsening after prolonged standing and at the end of the day, with relief upon leg elevation\u2014directly reflect the pathophysiology of venous stasis. Furthermore, the patient's demographic and lifestyle factors, including her age (58), female sex, and overweight status, are all well-established and significant risk factors for the development of CVI.\n\n    However, CVI alone does not adequately explain the patient's two most distressing symptoms: overwhelming fatigue and unusual bruising. The presence of spontaneous ecchymoses is a significant red flag for a potential underlying hematological issue, such as a platelet dysfunction or a disorder of the coagulation cascade (coagulopathy). The profound fatigue could be a symptom of an underlying anemia, which can be associated with bleeding disorders, or it could be a non-specific symptom of another systemic process. Therefore, the most probable working diagnosis is a dual pathology: clinically evident CVI that explains the leg symptoms, co-existing with a yet-to-be-identified systemic or hematological condition that is responsible for the bruising and fatigue.\n\n**Considered Differential Diagnoses:**\n\n*   **Deep Vein Thrombosis (DVT):** This was considered as a potential alternative or complicating factor. Leg swelling and pain are key features of DVT. However, the chronic nature of the patient's symptoms (developing over weeks) and the bilateral, though scattered, nature of her complaints make an acute DVT less likely as the sole explanation. Furthermore, the reported relief with leg elevation is more characteristic of CVI than an acute thrombotic event. Nevertheless, given the potential for DVT to lead to serious complications like pulmonary embolism and its role in causing post-thrombotic syndrome (a form of secondary CVI), it remains a critical differential that must be definitively ruled out.\n\n*   **Primary Bleeding Disorder / Platelet Dysfunction:** This was also evaluated as a primary diagnosis. The spontaneous bruising and significant fatigue are highly suggestive of a primary hematological condition, such as idiopathic thrombocytopenic purpura (ITP) or another coagulopathy. While this diagnosis would explain the bruising and potentially the fatigue (if associated with anemia), it fails to account for the classic venous stasis symptoms: the leg heaviness, end-of-day swelling, prominent veins, and relief with elevation. For this reason, a primary bleeding disorder is considered less likely to be the *sole* diagnosis, but it remains a very strong possibility as a co-existing condition that requires urgent investigation.\n\n### **4. Recommended Exams and Further Investigation**\nBased on the clinical uncertainty and the need to rule out critical differentials while confirming the provisional diagnosis, the following investigative steps are recommended. These tests are essential to transition the provisional diagnosis into a definitive one and to ensure patient safety.\n\n*   **Lower Extremity Venous Duplex Ultrasound:** This non-invasive imaging study is the cornerstone for evaluating the patient's leg symptoms. Its primary purpose is twofold: first, to assess for venous valve incompetence and reflux, which would confirm the diagnosis of Chronic Venous Insufficiency; and second, to definitively rule out the presence of a Deep Vein Thrombosis (DVT) in the lower extremities.\n\n*   **Complete Blood Count (CBC) with differential and platelet count:** This fundamental blood test is required to investigate the patient's fatigue and bruising. It will assess for anemia (low red blood cell count or hemoglobin) as a cause for her profound fatigue and will provide a platelet count. A low platelet count (thrombocytopenia) would be a direct and immediate explanation for her tendency to bruise easily.\n\n*   **Prothrombin Time (PT) and Activated Partial Thromboplastin Time (aPTT):** These coagulation studies are essential to evaluate the integrity of the extrinsic and intrinsic pathways of the blood clotting cascade. Abnormalities in these tests would point towards a specific coagulopathy (a deficiency or dysfunction of clotting factors) that could be responsible for the unusual bruising.\n\n*   **D-dimer:** This blood test measures a substance released when a blood clot breaks down. While not highly specific, a negative D-dimer result has a high negative predictive value, making it a very useful tool to help rule out an acute thrombotic process like DVT, thereby increasing confidence in an alternative diagnosis.\n\n### **5. Recommended Management Plan (Plan)**\nBased on the probable diagnosis of Chronic Venous Insufficiency with a suspected co-existing coagulopathy, the following comprehensive management plan is recommended. The immediate focus is on symptomatic relief, patient safety, and obtaining a definitive diagnosis through the investigations outlined above.\n\n**Patient-Specific Considerations:**\nIt is noted that the patient is overweight, which is a modifiable risk factor for CVI, and management should include counseling on weight management. Her occasional use of ibuprofen, a non-steroidal anti-inflammatory drug (NSAID), is a point of concern. NSAIDs can interfere with platelet function and potentially exacerbate bleeding or bruising. It is strongly recommended that she avoid or strictly limit the use of ibuprofen until the cause of her bruising is fully investigated and identified. The patient reports no known allergies, which allows for a wide range of diagnostic and therapeutic options.\n\n**1. Initial Management and Symptomatic Relief (Pending Diagnosis):**\n*   **Leg Elevation:** To manage the symptoms of CVI, the patient should be instructed to elevate her legs above the level of her heart for 15-20 minute periods, 3-4 times per day. The rationale for this intervention is to use gravity to assist venous return, thereby reducing venous hypertension and alleviating swelling and the sensation of heaviness.\n*   **Compression Therapy:** The use of graduated compression stockings is a cornerstone of CVI management. These stockings apply the greatest pressure at the ankle, with decreasing pressure up the leg, which mechanically assists in moving blood out of the legs and back toward the heart. She should be fitted for a pair (e.g., Class 1 or 2) and instructed to apply them in the morning before significant swelling occurs. *This recommendation is made with the caveat that an acute DVT must be ruled out first.*\n*   **Avoidance of Prolonged Static Postures:** The patient should be counseled to avoid long periods of uninterrupted sitting or standing. Her occupation as a librarian may require this, so she should be encouraged to take frequent short breaks to walk around, flex her ankles, and change positions to activate the calf muscle pump, which is critical for venous circulation.\n\n**2. Non-Pharmacological and Lifestyle Interventions:**\n*   **Regular Low-Impact Exercise:** A program of regular exercise, such as walking, swimming, or cycling, is highly recommended. The clinical evidence for this intervention shows that it improves circulation, strengthens the calf muscles, and can aid in weight management, all of which are beneficial for managing CVI.\n*   **Dietary Modification:** Counseling on a healthy, balanced diet that is low in fat and sugar and rich in fruits, vegetables, and whole grains should be provided. This supports overall vascular health and is a key component of any weight management strategy.\n*   **Trauma Avoidance:** Given the unexplained bruising, the patient should be advised to be particularly cautious to avoid bumps, falls, and other minor injuries that could lead to significant ecchymosis or hematoma formation.\n\n**3. Pharmacological Treatment (Conditional on Diagnosis):**\nIt must be emphasized that definitive pharmacological treatment is entirely contingent upon the results of the recommended diagnostic workup.\n*   **If DVT is Confirmed:** The patient would require immediate initiation of anticoagulation therapy. The rationale is to prevent the clot from growing and to reduce the risk of it embolizing to the lungs (pulmonary embolism). Treatment options include Low Molecular Weight Heparin (LMWH) or a Direct Oral Anticoagulant (DOAC) like rivaroxaban or apixaban.\n*   **If a Coagulopathy is Confirmed:** Management will be highly specific to the identified disorder. This would necessitate an urgent referral to a Hematologist for specialist evaluation and a tailored treatment plan, which could involve factor replacement, immunosuppressants, or other targeted therapies.\n*   **For CVI Symptoms:** If DVT and significant coagulopathy are ruled out, the primary management for CVI remains non-pharmacological. While some venoactive drugs (e.g., micronized purified flavonoid fractions) are used in some regions to improve venous tone and symptoms, their role is secondary to compression and lifestyle modification.\n\n### **6. Concluding Summary**\nIn summary, this 58-year-old female presents with a complex clinical picture characterized by classic symptoms of Chronic Venous Insufficiency in her lower extremities, complicated by the highly concerning systemic symptoms of overwhelming fatigue and unusual bruising. The provisional diagnosis is CVI with a suspected, co-existing hematological disorder or coagulopathy. The immediate priority is a comprehensive diagnostic workup, including a lower extremity venous duplex ultrasound and a full panel of hematological and coagulation studies, to confirm the CVI, definitively rule out a dangerous DVT, and identify the etiology of her bruising and fatigue. Initial management should focus on conservative, non-pharmacological measures to alleviate her leg symptoms while exercising caution regarding medications that could worsen bleeding. The final, long-term management plan will be dictated by the results of these crucial investigations and will likely require a multidisciplinary approach, potentially involving consultation with both Hematology and Vascular Medicine specialists.\n\n### References\n\n**Knowledge Base References:**\n- MedlinePlus XML documents (retrieved via `retrieve_documents` tool)\n- Venous insufficiency: MedlinePlus Medical Encyclopedia (https://medlineplus.gov/ency/article/000203.htm)\n- Deep Vein Thrombosis | DVT | MedlinePlus: https://medlineplus.gov/deepveinthrombosis.html\n- Chronic Venous Insufficiency | The Foundation to Advance Vascular Cures: https://www.vascularcures.org/chronic-venous-insufficiency-cvi\n- Deep vein thrombosis: MedlinePlus Medical Encyclopedia: https://medlineplus.gov/ency/article/000156.htm\n- Deep vein thrombosis (DVT) - Diagnosis & treatment - Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/deep-vein-thrombosis/diagnosis-treatment/drc-20352563\n- Varicose veins - Diagnosis and treatment - Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/varicose-veins/diagnosis-treatment/drc-20350649\n- Blood Thinners | Anticoagulants | MedlinePlus: https://medlineplus.gov/bloodthinners.html\n- MedlinePlus XML documents (various, retrieved via `retrieve_documents` tool).\n\n**PubMed Articles:**\n- Improving concordance with long-term compression therapy amongst people with venous hypertension and lower leg ulceration: A Delphi study- patient cohort. PMID: 40730126. URL: https://pubmed.ncbi.nlm.nih.gov/40730126/\n- Application of bacterial cellulose film as a wound dressing in varicose vein surgery: A randomized clinical trial. PMID: 40846177. URL: https://pubmed.ncbi.nlm.nih.gov/40846177/\n- A systematic review of anatomical reflux patterns in primary chronic venous disease. PMID: 39025298. URL: https://pubmed.ncbi.nlm.nih.gov/39025298/\n- Anticoagulant combined with antiplatelet therapy is associated with improved iliac vein stent patency following thrombolysis and thrombectomy of subacute iliofemoral deep vein thrombosis: A retrospective propensity score matching cohort study. PMID: 41185601. URL: https://pubmed.ncbi.nlm.nih.gov/41185601/\n- Short Versus Long Venous Thromboembolism Prophylaxis Following Elective Total Hip Arthroplasty: A Bayesian Network Meta-Analysis of Efficacy and Safety. PMID: 41177191. URL: https://pubmed.ncbi.nlm.nih.gov/41177191/\n- Chinese expert consensus on prevention, diagnosis, and management of venous thromboembolism in adult burn patients (2024). PMID: 41177896. URL: https://pubmed.ncbi.nlm.nih.gov/41177896/\n- Lupus anticoagulant-hypoprothrombinemia syndrome in children: Three case reports and systematic review of the literature. PMID: 37480550. URL: https://pubmed.ncbi.nlm.nih.gov/37480550/\n- Comparative efficacy of 19 drug therapies for patients with idiopathic thrombocytopenic purpura: a multiple-treatments network meta-analysis. PMID: 35149911. URL: https://pubmed.ncbi.nlm.nih.gov/35149911/\n- Neuralgia-inducing cavitational osteonecrosis - A systematic review. PMID: 33893686. URL: https://pubmed.ncbi.nlm.nih.gov/33893686/\n"""

    print(f"Evaluation text length: {len(evaluation_text)}")
    print(f"=== Evaluation Complete ===\n")
    
    return EvaluationResponse(evaluation=evaluation_text)
