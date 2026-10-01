from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

class EmotionalState(BaseModel):
    current_emotion: str = Field(description="The primary emotion of the user (e.g., anxiety, sadness, joy)")
    stressors: list[str] = Field(description="List of things causing stress or pressure")
    goals: list[str] = Field(description="The goals mentioned by the user")

# Προσθέτουμε το memory_store για να κρατάμε το ιστορικό του χρήστη
class AgentState(TypedDict):
    user_message: str
    extracted_data: dict
    memory_store: dict 

parser = JsonOutputParser(pydantic_object=EmotionalState)
llm = ChatOllama(model="llama3.1", temperature=0.1, format="json")

prompt = PromptTemplate(
    template="""You are an analytical AI. Extract the requested information from the user input strictly in JSON format.
    
FORMAT INSTRUCTIONS:
{format_instructions}

USER INPUT: 
{user_input}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extraction_chain = prompt | llm | parser

# Κόμβος 1: Εξαγωγή δεδομένων από το τρέχον μήνυμα
def extract_emotion_node(state: AgentState):
    print("--> Node 1: Extracting new data from message...")
    result = extraction_chain.invoke({"user_input": state["user_message"]})
    return {"extracted_data": result} 

# Κόμβος 2: Διαχείριση Μνήμης (Temporal Logic)
def memory_manager_node(state: AgentState):
    print("--> Node 2: Updating Temporal Memory Store...")
    extracted = state["extracted_data"]
    memory = state["memory_store"]

    # Ενημερώνουμε το τρέχον συναίσθημα (Overwrite)
    if "current_emotion" in extracted:
        memory["latest_emotion"] = extracted["current_emotion"]

    # Προσθέτουμε νέους στρεσογόνους παράγοντες χωρίς να χάνουμε τους παλιούς (Append)
    for stressor in extracted.get("stressors", []):
        if stressor not in memory["historical_stressors"]:
            memory["historical_stressors"].append(stressor)

    # Προσθέτουμε νέους στόχους (Append)
    for goal in extracted.get("goals", []):
        if goal not in memory["known_goals"]:
            memory["known_goals"].append(goal)

    return {"memory_store": memory}

# Στήσιμο του Graph με δύο κόμβους πλέον
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_emotion_node)
workflow.add_node("memory", memory_manager_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "memory") # Το γράφημα πάει από το extract στο memory
workflow.add_edge("memory", END)
app = workflow.compile()

if __name__ == "__main__":
    user_msg = "I'm completely overwhelmed. My thesis deadline is approaching, I feel like I have no time, and my only goal right now is to just get my degree so I can find a remote job."
    
    # Προσομοιώνουμε ότι ο χρήστης έχει ήδη μια παλιά μνήμη καταχωρημένη
    mock_existing_memory = {
        "latest_emotion": "calm",
        "historical_stressors": ["financial issues"],
        "known_goals": ["learn Python"]
    }
    
    print("Starting LangGraph workflow...\n")
    
    final_state = app.invoke({
        "user_message": user_msg, 
        "extracted_data": {},
        "memory_store": mock_existing_memory
    })
    
    print("\n--- Final Consolidated Memory Store ---")
    print(final_state["memory_store"])