import json
import os
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Persistent Storage Setup ---
MEMORY_FILE = "memory.json"

def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Default empty memory state if file doesn't exist
    return {
        "latest_emotion": "neutral",
        "historical_stressors": [],
        "known_goals": []
    }

def save_memory(memory_data: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=4)

# --- 2. Schemas & State ---
class EmotionalState(BaseModel):
    current_emotion: str = Field(description="The primary emotion of the user (e.g., anxiety, sadness, joy)")
    stressors: list[str] = Field(description="List of things causing stress or pressure")
    goals: list[str] = Field(description="The goals mentioned by the user")

class AgentState(TypedDict):
    user_message: str
    extracted_data: dict
    memory_store: dict
    agent_response: str  # New field to hold the final reply

# --- 3. LLMs & Chains ---
# Extractor LLM (Strict JSON)
extractor_llm = ChatOllama(model="llama3.1", temperature=0.1, format="json")
parser = JsonOutputParser(pydantic_object=EmotionalState)
extract_prompt = PromptTemplate(
    template="""You are an analytical AI. Extract the requested information from the user input strictly in JSON format.
    
FORMAT INSTRUCTIONS:
{format_instructions}

USER INPUT: 
{user_input}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
extraction_chain = extract_prompt | extractor_llm | parser

# Chat LLM (Creative, no JSON format restrictions)
chat_llm = ChatOllama(model="llama3.1", temperature=0.7)
chat_prompt = PromptTemplate(
    template="""You are an empathetic AI companion with excellent memory.
    
USER'S MEMORY PROFILE:
{memory_store}
    
INSTRUCTIONS:
Respond to the user's latest message naturally and supportively. Subtly incorporate your knowledge of their ongoing stressors and goals to show you remember their context. Keep the response concise.

USER'S LATEST MESSAGE: 
{user_message}""",
    input_variables=["memory_store", "user_message"]
)
chat_chain = chat_prompt | chat_llm | StrOutputParser()

# --- 4. Nodes ---
def extract_emotion_node(state: AgentState):
    print("--> Node 1: Extracting new data...")
    result = extraction_chain.invoke({"user_input": state["user_message"]})
    return {"extracted_data": result} 

def memory_manager_node(state: AgentState):
    print("--> Node 2: Consolidating Memory & Saving to Disk...")
    extracted = state["extracted_data"]
    memory = state["memory_store"]

    if "current_emotion" in extracted:
        memory["latest_emotion"] = extracted["current_emotion"]

    for stressor in extracted.get("stressors", []):
        if stressor not in memory["historical_stressors"]:
            memory["historical_stressors"].append(stressor)

    for goal in extracted.get("goals", []):
        if goal not in memory["known_goals"]:
            memory["known_goals"].append(goal)
            
    # Persist changes to JSON file
    save_memory(memory)
    
    return {"memory_store": memory}

def chat_node(state: AgentState):
    print("--> Node 3: Generating empathetic response...")
    response = chat_chain.invoke({
        "memory_store": json.dumps(state["memory_store"]),
        "user_message": state["user_message"]
    })
    return {"agent_response": response}

# --- 5. Graph Compilation ---
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_emotion_node)
workflow.add_node("memory", memory_manager_node)
workflow.add_node("chat", chat_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "memory")
workflow.add_edge("memory", "chat")
workflow.add_edge("chat", END)

app = workflow.compile()

# --- 6. Execution ---
if __name__ == "__main__":
    # Load permanent memory from disk
    current_memory = load_memory()
    
    user_msg = "I'm completely overwhelmed. My thesis deadline is approaching, I feel like I have no time, and my only goal right now is to just get my degree so I can find a remote job."
    
    print("Starting LangGraph workflow...\n")
    
    final_state = app.invoke({
        "user_message": user_msg, 
        "extracted_data": {},
        "memory_store": current_memory,
        "agent_response": ""
    })
    
    print("\n===============================")
    print("🤖 AGENT RESPONSE:")
    print(final_state["agent_response"])
    print("===============================\n")