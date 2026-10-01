from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# 1. Pydantic Schema (Defines the expected output structure)
class EmotionalState(BaseModel):
    current_emotion: str = Field(description="The primary emotion of the user (e.g., anxiety, sadness, joy)")
    stressors: list[str] = Field(description="List of things causing stress or pressure")
    goals: list[str] = Field(description="The goals mentioned by the user")

# 2. Define Agent State (The working memory of the graph during execution)
class AgentState(TypedDict):
    user_message: str
    extracted_data: dict

# 3. LLM and Chain Setup
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

# 4. Define the Extraction Node
def extract_emotion_node(state: AgentState):
    print("--> Executing Node: Emotion & Data Extraction...")
    
    # Process the user message from the current state
    result = extraction_chain.invoke({"user_input": state["user_message"]})
    
    # Return the updated data to the state
    return {"extracted_data": result} 

# 5. LangGraph Setup and Compilation
workflow = StateGraph(AgentState)
workflow.add_node("extract", extract_emotion_node)
workflow.set_entry_point("extract")
workflow.add_edge("extract", END)
app = workflow.compile()

# 6. Main Execution
if __name__ == "__main__":
    user_msg = "I'm completely overwhelmed. My thesis deadline is approaching, I feel like I have no time, and my only goal right now is to just get my degree so I can find a remote job."
    
    print("Starting LangGraph workflow...\n")
    
    # Initialize the graph with the user's message
    final_state = app.invoke({"user_message": user_msg, "extracted_data": {}})
    
    print("\n--- Final Extracted Data ---")
    print(final_state["extracted_data"])