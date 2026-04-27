import streamlit as st
import json
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import os

# Try to import Groq (for cloud deployment)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SciRAG - scRNA & ATAC Assistant",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS 
st.markdown("""
    <style>
    /* Only style the custom containers */
    .retrieved-passage {
        background-color: #2d3748;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
        font-family: inherit;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .answer-box {
        background-color: #2d3748;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
        font-family: inherit;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .stats-box {
        background-color: #f0f2f6;
        padding: 0.5rem 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
        border-left: 3px solid #6c63ff;
    }

    .stats-box .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e293b;
        margin: 0;
        line-height: 1.2;
    }

    .stats-box .stat-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #475569;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Make links/references inside answer box also white */
    .answer-box strong, .answer-box a {
        color: #ffffff;
    }
            
    .corpus-badge {
        background-color: #6c63ff;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        display: inline-block;
        margin-bottom: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False
if 'query' not in st.session_state:
    st.session_state.query = ""

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    """Check if running on Streamlit Cloud"""
    # Check for common Streamlit Cloud environment variables
    return (
        os.environ.get("STREAMLIT_CLOUD") == "true" or
        os.environ.get("IS_STREAMLIT_CLOUD") == "true" or
        os.environ.get("STREAMLIT_SHARING") == "true" or
        # Check if running on share.streamlit.io domain
        "share.streamlit.io" in os.environ.get("STREAMLIT_BROWSER_ADDRESS", "") or
        ".streamlit.app" in os.environ.get("STREAMLIT_BROWSER_ADDRESS", "")
    )

@st.cache_resource
def load_models():
    """Load BioBERT model and FAISS index"""
    try:
        if not os.path.exists("data/faiss_index.bin"):
            st.error("FAISS index not found. Please run the notebook first.")
            return None, None, None, None, None
        
        if not os.path.exists("data/embedded_data.json"):
            st.error("Embedded data not found. Please run the notebook first.")
            return None, None, None, None, None
        
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, use_safetensors=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        index = faiss.read_index("data/faiss_index.bin")
        
        with open("data/embedded_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return tokenizer, model, device, index, data
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def embed_query(tokenizer, model, device, text):
    """Generate embedding for a query"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :]
    embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy().astype("float32")

def retrieve_relevant_chunks(query, index, data, tokenizer, model, device, k=5):
    """Retrieve top-k relevant chunks for a query"""
    all_results = []
    q_embedding = embed_query(tokenizer, model, device, query)
    distances, indices = index.search(q_embedding, k*2)
    
    for idx, dist in zip(indices[0], distances[0]):
        all_results.append((int(idx), float(dist)))
    
    seen = set()
    unique_results = []
    for idx, dist in all_results:
        if idx not in seen:
            seen.add(idx)
            unique_results.append((idx, dist))
    
    return unique_results[:k]

def generate_answer_groq(query, retrieved_chunks, references, api_key):
    """Generate answer using Groq API"""
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a scientific assistant. Answer using ONLY the provided context.

Context:
{context}

Question: {query}

Instructions:
- If the context contains the answer, provide it directly and concisely
- If the answer is not found, say exactly: "Based on the provided papers, this information was not found."
- Use bullet points for multiple points
- Do not add conversational phrases

Answer:"""
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # More stable model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        
        # If answer is too short or indicates not found
        if len(answer) < 20 or "not found" in answer.lower():
            ref_text = "\n\n**Note:** The retrieved passages did not contain relevant information.\n\n**Retrieved from:**\n" + "\n".join([f"- {r}" for r in references])
        else:
            ref_text = "\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])
        
        return answer + ref_text
    
    except Exception as e:
        return f"⚠️ API Error: {str(e)[:200]}\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])
    
def generate_answer_ollama(query, retrieved_chunks, references):
    """Generate answer using Ollama (local)"""
    try:
        import ollama
    except ImportError:
        return "⚠️ Ollama not installed. Please install ollama or use cloud deployment.\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])
    
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a scientific assistant. Answer using ONLY the provided context.

Rules:
- Start directly with the answer
- No conversational phrases
- Keep it concise and factual
- Use bullet points if multiple points
- If not found, say: "Not found in the provided papers"

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
        
        answer = response["message"]["content"].strip()
        ref_text = "\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])
        
        return answer + ref_text
    
    except Exception as e:
        return f"Error with Ollama: {str(e)}\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])

def main():
    # Detect if running on Streamlit Cloud
    cloud_mode = is_streamlit_cloud()
    
    # Get Groq API key from secrets (cloud) or environment (local test)
    groq_api_key = None
    use_groq = False
    
    if GROQ_AVAILABLE:
        try:
            # Try to get from Streamlit secrets
            groq_api_key = st.secrets.get("GROQ_API_KEY")
            if groq_api_key:
                use_groq = True
        except:
            pass
        
        # If not in secrets, try environment variable
        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if groq_api_key:
                use_groq = True
    
    # Force use Groq on cloud if available
    if cloud_mode and not use_groq:
        st.warning("⚠️ Cloud mode detected but no Groq API key found. Add it in Secrets (Settings → Secrets).")
    
    # Header
    st.markdown('<div style="font-size: 3rem; text-align: center;">🧬 SciRAG</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 1.8rem;">Retrieval-Augmented Generation for scRNA-seq & ATAC-seq | BioBERT + Gemma</div>', unsafe_allow_html=True)
    
    # Show mode indicator
    if use_groq:
        st.success("🚀 Cloud Mode: Using Groq API (Llama 3.1 70B)")
    elif cloud_mode and not use_groq:
        st.warning("⚠️ Add Groq API key in Secrets (Settings → Secrets) to enable AI answers.")
    else:
        st.info("💻 Local Mode: Using Ollama (Gemma 2B)")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 About the Corpus")
        st.markdown("""
        - **Topic**: Single-cell RNA sequencing & ATAC-seq
        - **Source**: PubMed (free full-text papers)
        - **Documents**: 15 scientific papers
        - **Embedding Model**: BioBERT (768-dim)
        - **LLM**: Groq (Llama 3.1) / Ollama (Gemma 2B)
        - **Vector DB**: FAISS (cosine similarity)
        """)
        
        st.markdown("---")
        st.markdown("## ❓ Example Questions (Click to Ask)")
        
        example_queries = [
            "What is single cell RNA sequencing?",
            "How does scRNA-seq help in cancer research?",
            "What are the limitations of single-cell sequencing?",
            "What techniques are used in scRNA-seq?",
            "Applications of scRNA-seq in tumor analysis"
        ]
        
        for q in example_queries:
            if st.button(f"🔬 {q}", key=q, use_container_width=True):
                st.session_state.query = q
                st.session_state.trigger_search = True
                st.rerun()
        
        st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "**Ask a question about scRNA-seq, ATAC-seq, or cancer biology.**",
            placeholder="Example: What are the applications of single-cell sequencing in tumor analysis?",
            height=100,
            key="query_input",
            value=st.session_state.get("query", "")
        )
           
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        
        with col_btn2:
            clear = st.button("🗑️ Clear", use_container_width=True)
            if clear:
                st.session_state.query = ""
                st.session_state.trigger_search = False
                st.rerun()
        
        should_search = submit or (st.session_state.get("trigger_search", False))
        
        if should_search and query:
            with st.spinner("🔍 Retrieving relevant passages..."):
                tokenizer, model, device, index, data = load_models()
                
                if tokenizer is None:
                    st.session_state.trigger_search = False
                    return
                
                results = retrieve_relevant_chunks(query, index, data, tokenizer, model, device, k=5)
                
                if len(results) == 0:
                    st.warning("No relevant passages found. Try a different question.")
                    st.session_state.trigger_search = False
                    return
                
                retrieved_chunks = []
                references = set()
                retrieved_details = []
                
                for idx, score in results:
                    chunk = data[idx]
                    retrieved_chunks.append(chunk["text"])
                    references.add(chunk["doc_id"])
                    retrieved_details.append({
                        "doc_id": chunk["doc_id"],
                        "passage_id": chunk["passage_id"],
                        "score": score,
                        "text": chunk["text"]
                    })
                
                # Generate answer - choose backend
                with st.spinner("🤖 Generating answer..."):
                    if use_groq and groq_api_key:
                        answer = generate_answer_groq(query, retrieved_chunks, references, groq_api_key)
                    elif cloud_mode and not use_groq:
                        answer = "⚠️ **Groq API key required for cloud deployment.**\n\nPlease add your Groq API key in Streamlit Secrets:\n1. Go to your app dashboard\n2. Click 'Manage app' → 'Secrets'\n3. Add: `GROQ_API_KEY = \"your_key_here\"`\n\n**References:**\n" + "\n".join([f"- {r}" for r in references])
                    else:
                        # Try local Ollama
                        answer = generate_answer_ollama(query, retrieved_chunks, references)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Display retrieved passages
                with st.expander("📖 View Retrieved Passages (Top-5)", expanded=True):
                    st.caption("These passages were retrieved from the corpus and used to generate the answer above.")
                    
                    for i, detail in enumerate(retrieved_details, 1):
                        st.progress(min(1.0, detail['score']), text=f"Relevance: {detail['score']:.3f}")
                        
                        st.markdown(f"""
                        <div class="retrieved-passage">
                            <strong>Rank #{i}</strong><br>
                            <strong>Document:</strong> {detail['doc_id']}<br>
                            <strong>Passage ID:</strong> {detail['passage_id']}<br>
                            <strong>Score:</strong> {detail['score']:.4f}<br><br>
                            <strong>Content:</strong><br>
                            {detail['text'][:800]}...
                        </div>
                        """, unsafe_allow_html=True)
                
                with st.expander("📊 Retrieval Statistics"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Passages Retrieved", len(results))
                    with col_b:
                        st.metric("Unique Documents", len(references))
                    with col_c:
                        st.metric("Avg Relevance", f"{np.mean([d['score'] for d in retrieved_details]):.3f}")
                
                st.session_state.trigger_search = False
        
        elif should_search and not query:
            st.warning("Please enter a question or click an example.")
            st.session_state.trigger_search = False
    
    with col2:
        st.markdown("##### 📊 Corpus Stats")
        
        # Stat 1
        st.markdown("""
        <div class="stats-box">
            <div class="stat-label">📄 Total Papers</div>
            <div class="stat-number">15</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stat 2
        st.markdown("""
        <div class="stats-box">
            <div class="stat-label">📝 Total Passages</div>
            <div class="stat-number">681</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stat 3
        st.markdown("""
        <div class="stats-box">
            <div class="stat-label">🔢 Embedding Dim</div>
            <div class="stat-number">768</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stat 4
        st.markdown("""
        <div class="stats-box">
            <div class="stat-label">📏 Chunk Size</div>
            <div class="stat-number">300 tokens</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("##### 💡 Note")
        st.info("All answers are generated **exclusively** from the 15 papers in the database. No external information is used.")
        
if __name__ == "__main__":
    if 'trigger_search' not in st.session_state:
        st.session_state.trigger_search = False
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    main()