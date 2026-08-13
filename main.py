
import os
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv()


# Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered Health Symptom Assistant",
    page_icon="👨‍⚕️",
    layout="centered"
)


# Embedding class using Sentence Transformer
class HuggingFaceEmbeddings(Embeddings):

    # Load the embedding model
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # Create embeddings for the document chunks
    def embed_documents(self, texts):
        return [
            self.model.encode(text).tolist()
            for text in texts
        ]

    # Create an embedding for the user query
    def embed_query(self, text):
        return self.model.encode(text).tolist()


# Custom LangChain class for Gemini
class GeminiLLM(LLM):

    model: str = "gemini-3.5-flash"
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Send the prompt to Gemini
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env"
            )

        client = genai.Client(
            api_key=self.api_key
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if response.text:
            return response.text

        return "No response was generated."

    # Return the Gemini model information
    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    # Return the LLM type
    @property
    def _llm_type(self) -> str:
        return "gemini"


# Load the embedding model only once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# Load the FAISS vector database
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    return FAISS.load_local(
        "medical_vectordb",
        embeddings,
        allow_dangerous_deserialization=True
    )


# Combine the retrieved documents into one context
def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# Retrieve relevant documents from FAISS and generate an answer
def rag_answer(question):

    # Get the top 4 documents along with their similarity scores
    results = vectorstore.similarity_search_with_score(
        question,
        k=4
    )

    if not results:
        return (
            "No relevant medical information was found.",
            [],
            None
        )

    # FAISS returns distance scores.
    # Lower distance means the result is more similar.
    best_score = results[0][1]

    print(
        f"Best FAISS similarity distance: {best_score}"
    )

    # This is a starting threshold.
    # Test different queries and adjust it based on the scores.
    SIMILARITY_THRESHOLD = 1.0

    # Reject the query when the closest result is not relevant enough
    if best_score > SIMILARITY_THRESHOLD:
        return (
            "I could not find sufficiently relevant "
            "medical information for your question.",
            [],
            best_score
        )

    # Extract the documents from the search results
    docs = [
        doc
        for doc, score in results
    ]

    # Create the context for Gemini
    context = format_docs(docs)

    # Add the retrieved context and user question to the prompt
    prompt = prompt_template.format(
        context=context,
        question=question
    )

    # Generate the final response using Gemini
    answer = llm.invoke(prompt)

    return str(answer), docs, best_score


# Application title
st.title(
    "👨‍⚕️ AI-Powered Health Symptom Assistant"
)

# Display the medical disclaimer
st.warning(
    "This is an educational prototype and is not a substitute "
    "for professional medical advice. Always consult a doctor."
)


# Load the vector database and Gemini
try:
    vectorstore = load_vectorstore()
    llm = GeminiLLM()

except Exception as e:
    st.error(
        f"Error loading application: {e}"
    )
    st.stop()


# Prompt used by Gemini to generate the response
prompt_template = PromptTemplate(
    template="""
You are a careful health information assistant.

Use the medical knowledge context below to answer
the user's question.

Do not provide a definitive diagnosis.

Consider:
- Symptoms
- Duration
- Severity

Provide:
1. Possible explanations or differential diagnoses.
2. Important warning signs or red flags.
3. General next steps.

Important rules:
- Use the retrieved medical context as the primary source.
- Do not force unrelated context to fit the user's question.
- Do not interpret ordinary statements or emotions as symptoms.
- Do not invent medical information.
- If the context does not contain enough information,
  clearly say so.
- Do not claim certainty.
- Always include:
  "Disclaimer: Consult a doctor."

Medical Knowledge Context:
{context}

User Information:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# Get the user's symptoms
symptoms = st.text_area(
    "What symptoms are you experiencing?",
    placeholder="Example: Fever, cough, headache and sore throat",
    height=100
)


# Get the duration of the symptoms
duration = st.text_input(
    "How long have you been experiencing these symptoms?",
    placeholder="Example: 3 days"
)


# Get the severity of the symptoms
severity = st.selectbox(
    "How severe are your symptoms?",
    [
        "Select severity",
        "Low",
        "Medium",
        "High"
    ]
)


# Create buttons
col1, col2 = st.columns(2)

with col1:
    get_diagnosis = st.button(
        "🔍 Get Health Information",
        type="primary",
        use_container_width=True
    )

with col2:
    exit_app = st.button(
        "❌ Exit",
        use_container_width=True
    )


# Stop the application when Exit is clicked
if exit_app:
    st.info(
        "The session has been ended. "
        "You can close this browser tab."
    )
    st.stop()


# Process the user's request
if get_diagnosis:

    # Check whether the user entered symptoms
    if not symptoms.strip():
        st.warning(
            "Please describe your symptoms."
        )

    # Check whether the user entered duration
    elif not duration.strip():
        st.warning(
            "Please specify how long you have "
            "been experiencing these symptoms."
        )

    # Check whether the user selected severity
    elif severity == "Select severity":
        st.warning(
            "Please select the severity of your symptoms."
        )

    else:

        # Combine all patient information into one query
        patient_information = f"""
Symptoms: {symptoms}

Duration: {duration}

Severity: {severity}
"""

        with st.spinner(
            "Searching medical knowledge and generating response..."
        ):

            try:

                # Retrieve relevant documents and generate the answer
                answer, source_docs, score = rag_answer(
                    patient_information
                )

                # If no relevant documents were found
                if not source_docs:

                    st.warning(answer)

                    if score is not None:
                        st.info(
                            f"FAISS similarity distance: "
                            f"{score:.4f}"
                        )

                else:

                    # Display the generated answer
                    st.markdown("### Response")
                    st.write(answer)

                    # Display the similarity score for testing
                    st.markdown(
                        f"**Best FAISS similarity distance:** "
                        f"{score:.4f}"
                    )

                    # Get source page numbers
                    pages = [
                        doc.metadata.get(
                            "page",
                            "Unknown"
                        )
                        for doc in source_docs
                    ]

                    # Remove duplicate page numbers
                    unique_pages = list(
                        dict.fromkeys(pages)
                    )

                    if unique_pages:
                        st.markdown(
                            "**Source Pages:** "
                            + ", ".join(
                                map(
                                    str,
                                    unique_pages
                                )
                            )
                        )

            except Exception as e:
                st.error(
                    f"Error generating response: {e}"
                )


# Medical report upload is disabled for now.
# You can enable this section later if needed.

# import pymupdf as fitz

# uploaded_pdf = st.file_uploader(
#     "Upload medical report (PDF):",
#     type="pdf"
# )

# if uploaded_pdf:
#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_pdf.getbuffer())

#     with fitz.open("temp.pdf") as pdf:
#         for page in pdf:
#             text = page.get_text("text")
#
#             if text:
#                 pdf_text += text + " "
