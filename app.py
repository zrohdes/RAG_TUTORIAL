import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Explorer for Students", page_icon="🧠", layout="wide")

# --- LOAD EMBEDDING MODEL (CACHED) ---
@st.cache_resource
def load_embedder():
    # Lightweight, fast model ideal for classroom demos
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- SAMPLE UNIVERSITY KNOWLEDGE BASE ---
DEFAULT_DOC = """CS-402: Applied Machine Learning - Fall Syllabus
Instructor: Dr. Maya Lin | Office Hours: Tuesdays 2:00 PM - 4:00 PM in Turing Hall, Room 312.
Grading Policy: Homework Assignments (30%), Midterm Exam (25%), Final Capstone Project (35%), Class Participation (10%).
Late Policy: Every student gets 3 total 'grace days' for the semester. Once grace days are exhausted, late homework is penalized by 15% per day.
Midterm Exam Date: October 18th during regular lecture hours. No laptops allowed; one handwritten cheat sheet (front and back) is permitted.
Capstone Project: Teams of 2 to 4 students must build an end-to-end ML pipeline. Final presentations take place on December 10th."""

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ RAG Hyperparameters")
    st.markdown("Adjust these knobs to see how they affect retrieval quality:")
    
    chunk_size = st.slider("Chunk Size (words)", min_value=15, max_value=80, value=35, step=5)
    chunk_overlap = st.slider("Chunk Overlap (words)", min_value=0, max_value=20, value=5, step=5)
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=5, value=2)
    
    st.divider()
    st.caption("Built for CS/AI Classroom Demos • Uses `all-MiniLM-L6-v2`")

# --- HEADER ---
st.title("🧠 How RAG Works: An Interactive X-Ray")
st.markdown(
    "**Retrieval-Augmented Generation (RAG)** gives an LLM an *open-book exam* instead of forcing it to rely only on memorized training data. "
    "Walk through the 4 stages below to see what happens under the hood."
)

# --- TABS FOR EACH PIPELINE STAGE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Ingestion & Chunking",
    "2️⃣ Embeddings (Text → Math)",
    "3️⃣ Semantic Retrieval",
    "4️⃣ Augmented Generation"
])

# ==========================================
# TAB 1: CHUNKING
# ==========================================
with tab1:
    st.subheader("Step 1: Breaking Documents into Chunks")
    st.info("LLMs have context limits, and embedding whole textbooks dilutes meaning. We split documents into smaller, overlapping pieces.")
    
    doc_input = st.text_area("Knowledge Base Document (Edit me!)", value=DEFAULT_DOC, height=180)
    chunks = chunk_text(doc_input, chunk_size, chunk_overlap)
    
    st.markdown(f"**Result:** Split into **{len(chunks)} chunks** (Size: `{chunk_size}` words, Overlap: `{chunk_overlap}` words).")
    
    for idx, c in enumerate(chunks):
        with st.expander(f"Chunk #{idx + 1} ({len(c.split())} words)", expanded=True):
            st.write(c)

# ==========================================
# TAB 2: EMBEDDINGS
# ==========================================
with tab2:
    st.subheader("Step 2: Converting Text into Vector Embeddings")
    st.info("Computers can't compare meaning using raw strings. An embedding model converts each chunk into a list of numbers (a vector) representing its semantic meaning.")
    
    if chunks:
        chunk_embeddings = embedder.encode(chunks)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_idx = st.selectbox(
                "Inspect a chunk's vector:",
                range(len(chunks)),
                format_func=lambda i: f"Chunk #{i + 1}: {chunks[i][:35]}..."
            )
            st.metric("Vector Dimensions", f"{chunk_embeddings.shape[1]} floats")
            st.caption("Each chunk is mapped to a point in 384-dimensional space.")
            
        with col2:
            st.markdown(f"**First 15 numbers of Chunk #{selected_idx + 1}'s embedding:**")
            st.code(np.round(chunk_embeddings[selected_idx][:15], 4).tolist())
            st.bar_chart(chunk_embeddings[selected_idx][:40])

# ==========================================
# TAB 3: RETRIEVAL
# ==========================================
with tab3:
    st.subheader("Step 3: Finding Relevant Chunks via Cosine Similarity")
    st.info("When a student asks a question, we embed the question using the **exact same model** and measure the angle (Cosine Similarity) between the query vector and every chunk vector.")
    
    query = st.text_input(
        "Ask a question about the syllabus:",
        value="How many late days do I get and what is the penalty after?"
    )
    
    if query and chunks:
        query_embedding = embedder.encode([query])
        chunk_embeddings = embedder.encode(chunks)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            "Chunk ID": [f"Chunk #{i+1}" for i in range(len(chunks))],
            "Similarity Score": similarities,
            "Text": chunks
        }).sort_values(by="Similarity Score", ascending=False)
        
        st.markdown("### 📊 Similarity Ranking")
        st.bar_chart(results_df.set_index("Chunk ID")["Similarity Score"])
        
        st.markdown(f"### 🎯 Top-{top_k} Retrieved Chunks")
        top_chunks = results_df.head(top_k)
        for _, row in top_chunks.iterrows():
            st.success(f"**{row['Chunk ID']}** (Score: `{row['Similarity Score']:.4f}`)\n\n{row['Text']}")

# ==========================================
# TAB 4: GENERATION
# ==========================================
with tab4:
    st.subheader("Step 4: Prompt Augmentation & Generation")
    st.info("Finally, we stuff the retrieved chunks into a prompt template and instruct the LLM to answer *only* using those facts.")
    
    if query and chunks:
        retrieved_context = "\n\n---\n\n".join(top_chunks["Text"].tolist())
        
        final_prompt = f"""You are a helpful teaching assistant. Answer the student's question using ONLY the context provided below. If the answer is not in the context, say "I don't have enough information."

CONTEXT:
{retrieved_context}

STUDENT QUESTION:
{query}

ANSWER:"""

        col_no_rag, col_rag = st.columns(2)
        
        with col_no_rag:
            st.markdown("### ❌ Standard LLM (No RAG)")
            st.warning("**Simulated Response:**\n\n*\"Late policies vary by university, though many courses deduct 10% per day. Please check your syllabus or email your professor.\"*")
            st.caption("Why? The base LLM was never trained on Dr. Maya Lin's private Fall syllabus.")
            
        with col_rag:
            st.markdown("### ✅ RAG-Powered LLM")
            st.success("**Grounded Response:**\n\n*\"You get 3 total grace days for the semester. Once those are exhausted, late homework is penalized by 15% per day.\"*")
            st.caption("Why? The retrieved context gave the LLM the exact facts it needed.")
            
        st.divider()
        st.markdown("### 🔍 Under the Hood: The Exact Prompt Sent to the LLM")
        st.code(final_prompt, language="markdown")
