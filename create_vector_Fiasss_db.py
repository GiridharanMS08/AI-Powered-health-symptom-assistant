from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings  
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pymupdf as fitz  # PyMuPDF
import os

# Proper HuggingFaceEmbeddings 
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in tqdm(texts, desc="Embedding documents")]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Process large PDF with progress bar
def process_large_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    documents = []
    try:
        with fitz.open(pdf_path) as pdf:
            total_pages = pdf.page_count
            for i in tqdm(range(total_pages), desc="Loading PDF pages", total=total_pages):
                try:
                    page = pdf[i]
                    text = page.get_text("text")
                    if text:
                        documents.append(Document(page_content=text, metadata={"page": i + 1}))
                    else:
                        print(f"Skipping page {i+1}: No text extracted")
                except Exception as e:
                    print(f"Error processing page {i+1}: {e}")
                    continue
    except Exception as e:
        print(f"Error opening PDF: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    medical_docs = []
    for doc in tqdm(documents, desc="Chunking PDF pages"):
        chunks = text_splitter.split_documents([doc])
        medical_docs.extend(chunks)

    return medical_docs

# Create and save knowledge base
def create_knowledge_base(pdf_path=r"..\AI-health-Assistant\data\medical.pdf"):
    if not os.path.exists("medical_vectordb"):
        print("Creating knowledge base from meical PDF...")
        medical_docs = process_large_pdf(pdf_path)

        embeddings = HuggingFaceEmbeddings()
        print("Generating embeddings...")
        try:
            vectorstore = FAISS.from_documents(medical_docs, embeddings)
            vectorstore.save_local("medical_vectordb")
            print("Knowledge base created and saved to 'medical_vectordb'.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
    else:
        print("Knowledge base already exists at 'medical'.")

if __name__ == "__main__":
    pdf_path = r"...\AI-health-Assistant\data\medical.pdf"
    try:
        create_knowledge_base(pdf_path)
    except Exception as e:
        print(f"Failed to create knowledge base: {e}")
