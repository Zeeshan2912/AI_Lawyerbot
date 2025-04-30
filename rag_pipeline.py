import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaLLM  # Updated import
from vector_database import faiss_db
from langchain.prompts import ChatPromptTemplate

# Step 1: Setup LLM (Using Ollama with a local model)
llm_model = OllamaLLM(
    model="llama2",  # Use a model you've pulled (e.g., "llama2", "mistral")
    temperature=0.0,
    system="You are a legal expert. Provide accurate, concise answers to legal questions."
)

# Step 2: Retrieve Docs (unchanged)
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 3: Answer Question (unchanged)
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

# Example usage
#question = "If a government forbids the right to assemble peacefully which articles are violated and why?"
#retrieved_docs = retrieve_docs(question)
#print("AI Lawyer: ", answer_query(documents=retrieved_docs, model=llm_model, query=question))