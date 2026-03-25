import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---- PAGE SETUP ----
st.set_page_config(page_title="StuddyBuddy AI", page_icon="🧠")
st.title("🧠 StudyBuddyAI")
st.subheader("Upload a pdf and chat with it")

# ---- FUNCTION 1: READ PDF ----
def get_pdf_text(pdf_file):
    text=""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---- FUNCTION 2: CHUNK TEXT ----
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# ---- FUNCTION 3: CREATE EMBEDDINGS ----
def get_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings  = model.encode(chunks)
    return embeddings

# ---- FUNCTION 4: BUILD FAISS INDEX ----
def build_faiss_index(embeddings):
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index

# ---- FUNCTION 5: SEARCH FAISS ----
def search_chunks(question, chunks, index):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([question])
    question_array = np.array(question_embedding).astype('float32')
    distances, indices = index.search(question_array, k=2)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": distances[0][i]
        })
    return results




# ---- PDF UPLOAD ----
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Step 1: Read PDF
    raw_text = get_pdf_text(uploaded_file)
    st.success(f"✅ PDF loaded! {len(raw_text)} characters extracted.")

    # Step 2: Chunk text
    chunks = get_chunks(raw_text)
    st.info(f"📦 Split into {len(chunks)} chunks.")

    # Step 3: Create embeddings
    embeddings = get_embeddings(chunks)
    st.info(f"🔢 Each chunk converted to {len(embeddings[0])} numbers.")

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)
    st.info(f"🗄️ Stored {index.ntotal} chunks in FAISS database.")

     # Step 5: Search box
    st.subheader("🔍 Search Your PDF:")
    question = st.text_input("Ask a question about your PDF:")

    if question:
        results = search_chunks(question, chunks, index)
        st.subheader("📌 Most Relevant Chunks Found:")
        for i, result in enumerate(results):
            st.write(f"**Result {i+1} — Distance: {result['distance']:.4f}**")
            st.write(result['chunk'])
            st.divider()
