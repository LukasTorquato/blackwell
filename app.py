from blackwell.utils import print_state_messages
from blackwell.anamnesis import *
from blackwell.evaluator import *
from blackwell.db_handler import ChromaDBHandler


def anamnesis_procedure(config):
    """
    Execute the anamnesis procedure conversation.

    Args:
        config: The configuration object containing the thread ID

    Returns:
        The anamnesis report generated from the anamnesis procedure
    """
    
    print(f"\nHi, I am your Clinical Evaluation AI. How can I assist you today?")
    query = input("Exit with 'e': ")
    # Initialize state with the query
    initial_state = {"more_research": False, "messages": [HumanMessage(content=query)]}
    result = AnamnesisAgent.invoke(initial_state, config)

    while result["messages"][-1].content.startswith("[ANAMNESIS REPORT]:") == False:
        query = input("Exit with 'e': ")
        if query.lower() == 'e':
            print("Exiting anamnesis procedure....")
            break
        # Run the graph
        result = AnamnesisAgent.invoke(initial_state, config)
        print_state_messages(result)
    report = result["messages"][-1].content
    print(f"\nFinal Anamnesis Report:\n{report}")

    return report


def evaluation_procedure(report, config):
    """
    Execute the evaluation procedure utilizing the anamnesis report and the RAG pipeline.

    Args:
        config: The configuration object containing the thread ID

    Returns:
        The final medical evaluation generated from the evaluation procedure
    """
    
    print(f"\nEvaluating the anamnesis report...")
    initial_state = {"context": [], "anamnesis_report": HumanMessage(content=report), "query": None, "final_report": None}
    result = EvaluatorAgent.invoke(initial_state, config)
    #result["final_report"].pretty_print() 

    #print_state_messages(result)
    evaluation = result["final_report"]
    print(f"\nFinal Medical Evaluation:\n{evaluation}")

    return evaluation

def main():
    config = {"configurable": {"thread_id": 4}}
    #report = anamnesis_procedure(config)
    report = """[ANAMNESIS REPORT]:

        **Chief Complaint (CC):** Reddish, flat, sometimes scaly rash located above and sometimes on the penis, present for one week.

        **History of Present Illness (HPI):**
        *   **Onset:** Approximately one week ago.
        *   **Location/Character:** Reddish, flat, sometimes scaly rash located above and sometimes on the penis.
        *   **Symptoms:** Itching and discomfort, rated up to 5/10 after work.
        *   **Aggravating Factors:** Symptoms worsen due to sweating and friction from clothing (related to work as a restaurant floor staff), friction during sexual activity, and not wearing underwear.
        *   **Alleviating Factors:** Symptoms improve with the application of betamethasone/calcipotriol cream.
        *   **Timing:** Worse after work.

        **Past Medical History (PMH):**
        *   No prior history of this specific rash.
        *   No diagnosed medical conditions that the patient believes are related.

        **Medications and Allergies:**
        *   **Current Medications:** Multivitamin and Creatine.
        *   **Topical Treatment:** Betamethasone/calcipotriol cream (used for the rash).
        *   **Allergies:** Not specified, but no known pharmaceutical drugs are being taken.

        **Review of Systems (ROS):**
        *   Negative for fever, chills, joint pain, or changes in weight."""
    evaluation = evaluation_procedure(report, config)


if __name__ == "__main__":
    main()
