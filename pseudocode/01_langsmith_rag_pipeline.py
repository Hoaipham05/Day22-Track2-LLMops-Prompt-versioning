"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable


load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def _get_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


LANGSMITH_PROJECT = _get_env("LANGCHAIN_PROJECT", default="day22-langsmith-lab")
LANGSMITH_API_KEY = _get_env("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "LLM_API_KEY")
OPENAI_BASE_URL = _get_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL")
LLM_MODEL = _get_env("OPENAI_MODEL", "LLM_MODEL", default="gpt-5.4-mini")
EMBEDDING_MODEL = _get_env("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer the user's question. "
    "If the context does not contain the answer, say you do not have enough information.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _load_knowledge_base_text() -> str:
    candidates = [
        Path("data/knowledge_base.txt"),
        Path("pseudocode/data/knowledge_base.txt"),
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")

    fallback_sections = [
        "Machine learning includes supervised learning, unsupervised learning, and reinforcement learning.",
        "Overfitting happens when a model memorizes training data and performs poorly on new examples.",
        "The bias-variance tradeoff balances underfitting and overfitting.",
        "Regularization such as L1 and L2 adds penalties that reduce model complexity.",
        "Cross-validation evaluates a model by splitting data into folds and rotating the validation set.",
        "Backpropagation computes gradients with the chain rule to update neural network weights.",
        "CNNs are primarily used for image and other grid-like data.",
        "LSTMs use gates to preserve useful information and reduce vanishing gradients.",
        "Common activation functions include ReLU, sigmoid, and tanh.",
        "Pooling layers reduce spatial dimensions while keeping important features.",
        "Transformers use self-attention and process sequences in parallel.",
        "Word embeddings represent words as dense vectors with semantic relationships.",
        "Transfer learning pretrains a model on large corpora and fine-tunes it for a target task.",
        "BERT uses bidirectional transformer training with masked language modeling.",
        "Self-attention lets a model weigh token relationships across a sequence.",
        "GPT is trained autoregressively to predict the next token.",
        "Instruction tuning improves alignment with human instructions.",
        "RLHF aligns model behavior using human preference feedback.",
        "Chain-of-thought prompting encourages step-by-step reasoning.",
        "GPT-4 supports up to 128K tokens of context.",
        "RAG combines retrieval with generation so answers are grounded in external knowledge.",
        "A RAG pipeline usually has a retriever and an LLM generator.",
        "Dense retrieval uses embeddings and similarity search to find relevant passages.",
        "Chunking strategy affects retrieval quality and context window fit.",
        "Advanced RAG techniques include reranking, query expansion, HyDE, and iterative retrieval.",
        "Vector databases store and search high-dimensional embeddings.",
        "FAISS is a library for efficient similarity search.",
        "Text embeddings map text into vectors where similar meanings are close together.",
        "HNSW is a graph-based approximate nearest neighbor search method.",
        "Hybrid search combines dense vector search with sparse keyword search.",
        "LangChain is a framework for building LLM applications.",
        "LCEL is LangChain's pipe-based composition language for chains.",
        "LangGraph adds stateful graph-based orchestration to LangChain.",
        "LangChain supports buffer, summary, window, and vector-store memory patterns.",
        "Retrievers fetch relevant documents for a query from a data source.",
        "LangSmith provides tracing, debugging, testing, and monitoring for LLM apps.",
        "LangSmith traces capture inputs, outputs, latency, token usage, and errors.",
        "Prompt Hub stores, versions, and shares prompt templates.",
        "LangSmith helps monitor production applications with latency, errors, and feedback data.",
        "Datasets in LangSmith are used for systematic evaluation and comparison.",
        "RAGAS is an evaluation framework for RAG pipelines.",
        "Faithfulness checks whether answer claims are supported by retrieved context.",
        "Answer relevancy measures whether the answer addresses the original question.",
        "Context recall checks how much relevant reference information appears in retrieved context.",
        "RAGAS evaluation needs questions, answers, retrieved contexts, and optional references.",
        "Guardrails AI adds validation and safety checks to LLM outputs.",
        "PII includes emails, phone numbers, SSNs, and credit card numbers.",
        "Structured output validation makes sure responses match a schema like JSON.",
        "Constitutional AI uses guiding principles and self-critique to improve responses.",
        "Common AI safety issues include hallucination, toxicity, bias, PII leakage, and jailbreaking.",
    ]
    return "\n\n".join(f"Topic {i + 1}: {section}" for i, section in enumerate(fallback_sections))


def _make_llm() -> ChatOpenAI:
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(model=LLM_MODEL, **client_kwargs)


def _make_embeddings() -> OpenAIEmbeddings:
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, **client_kwargs)

# ── 1. Environment setup ────────────────────────────────────────────────────
# TODO: load your .env file using python-dotenv
# from dotenv import load_dotenv
# load_dotenv(...)

# TODO: set LangSmith environment variables BEFORE importing LangChain
# os.environ["LANGCHAIN_TRACING_V2"]  = "true"
# os.environ["LANGCHAIN_API_KEY"]     = "<your-langsmith-api-key>"
# os.environ["LANGCHAIN_PROJECT"]     = "<your-project-name>"
# os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
# TODO: import the libraries you need, for example:
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
# TODO: create a ChatOpenAI instance pointing to your endpoint
# llm = ChatOpenAI(
#     model=...,
#     api_key=...,
#     base_url=...,
# )

# TODO: create an OpenAIEmbeddings instance
# embeddings = OpenAIEmbeddings(
#     model=...,
#     api_key=...,
#     base_url=...,
# )


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Steps:
      a) Read your dataset
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Call FAISS.from_texts(chunks, embeddings) to build the index
      d) Return the vectorstore
    """
    text = _load_knowledge_base_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    embeddings = _make_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
# TODO: define a ChatPromptTemplate with:
#   - system message: instruct the LLM to answer using ONLY the provided context
#   - human message: the user's {question}
#
# RAG_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
#     ("human",  "{question}"),
# ])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = _make_llm()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


# ── 8. Sample questions (50 total — one per topic area) ────────────────────
SAMPLE_QUESTIONS = [
    "What are the three main types of machine learning?",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "How does regularization prevent overfitting?",
    "What is cross-validation?",
    "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?",
    "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?",
    "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?",
    "What are word embeddings?",
    "What is transfer learning in NLP?",
    "How does BERT handle language understanding?",
    "What is self-attention in transformers?",
    "What is GPT and how is it trained?",
    "What is instruction tuning?",
    "What is RLHF?",
    "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?",
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
    "What is dense retrieval?",
    "Why is chunking strategy important in RAG?",
    "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?",
    "What is FAISS?",
    "How do text embeddings capture semantic meaning?",
    "What is HNSW?",
    "What is hybrid search in vector databases?",
    "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?",
    "What is LangGraph?",
    "What memory types does LangChain support?",
    "What are LangChain retrievers?",
    "What is LangSmith?",
    "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?",
    "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?",
    "What is RAGAS?",
    "How does RAGAS compute faithfulness?",
    "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?",
    "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?",
    "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?",
    "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}\n")

    print(f"{len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
