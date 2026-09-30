from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Το Schema (ίδιο με πριν)
class EmotionalState(BaseModel):
    current_emotion: str = Field(description="Το βασικό συναίσθημα του χρήστη (π.χ. άγχος, θλίψη, χαρά)")
    stressors: list[str] = Field(description="Λίστα με τα πράγματα που του προκαλούν άγχος ή πίεση")
    goals: list[str] = Field(description="Οι στόχοι που αναφέρει ο χρήστης")

# 2. Ορίζουμε τον Parser που "μεταφράζει" την απάντηση σε JSON
parser = JsonOutputParser(pydantic_object=EmotionalState)

# 3. Φτιάχνουμε ένα καθαρό Prompt που δίνει ρητές οδηγίες
prompt = PromptTemplate(
    template="""Είσαι ένας βοηθός AI. Ανάλυσε το παρακάτω κείμενο και εξήγαγε τις πληροφορίες αυστηρά σε μορφή JSON.
    
ΟΔΗΓΙΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ:
{format_instructions}

ΚΕΙΜΕΝΟ ΧΡΗΣΤΗ: 
{user_input}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Αρχικοποιούμε το Llama. Το format="json" το "αναγκάζει" να βγάλει μόνο κώδικα.
# Βάζουμε θερμοκρασία 0.1 αντί για 0, για να αποφύγουμε λούπες.
llm = ChatOllama(model="llama3.1", temperature=0.1, format="json")

# 5. Ενώνουμε τα κομμάτια (Chain)
chain = prompt | llm | parser

print("Αναλύω το μήνυμα με το ασφαλές JSON σύστημα...")

user_message = "Έχω πελαγώσει τελείως. Η προθεσμία για τη διπλωματική πλησιάζει, νιώθω ότι δεν προλαβαίνω τίποτα και ο μόνος μου στόχος είναι να πάρω επιτέλους το πτυχίο για να βρω μια remote δουλειά."

# 6. Εκτέλεση
try:
    result = chain.invoke({"user_input": user_message})
    print("\n--- Αποτέλεσμα Εξαγωγής (JSON Parsing) ---")
    print(f"Συναίσθημα: {result.get('current_emotion')}")
    print(f"Στρεσογόνοι Παράγοντες: {result.get('stressors')}")
    print(f"Στόχοι: {result.get('goals')}")
    print("------------------------------------------")
except Exception as e:
    print(f"\n[ΣΦΑΛΜΑ]: {e}")