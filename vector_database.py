from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Upload & Load raw PDF(s)

pdfs_directory = 'pdfs/'

def upload_pdf():
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = 'universal_declaration_of_human_rights.pdf'
documents = load_pdf(file_path)
#print("PDF pages: ",len(documents))

# Step 2: Split documents into smaller chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

text_chunks = create_chunks(documents)
#print("Chunks count: ", len(text_chunks))

#Step 3: Setup Embeddings model (Use DeepSeek R1 with Ollama)
ollama_modelname = "deepseek-r1:1.5b"  # Remove the "ollama run" prefix
def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_modelname)
    return embeddings

#Step 4: Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_modelname))
faiss_db.save_local(FAISS_DB_PATH)