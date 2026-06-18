"""
ingest.py — RAG Recruitment Chatbot
Loads documents from /docs, chunks them, embeds with OpenAI,
and stores in ChromaDB.

Run once to build the vector database:
    python ingest.py

Re-run any time you update your documents in /docs.
"""
### -- Imports 

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

DOCS_DIR    = Path("docs")
CHROMA_DIR  = "chroma_db"
COLLECTION  = "recruitment"

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks to avoid splitting mid-answer

# ── Load documents ────────────────────────────────────────────────────────────

def load_documents(docs_dir: Path) -> list:
    documents = []
    supported = {".txt", ".md", ".pdf"}

    for file_path in sorted(docs_dir.iterdir()):
        if file_path.suffix.lower() not in supported:
            print(f"  Skipping {file_path.name} (unsupported type)")
            continue

        print(f"  Loading {file_path.name}...")

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        else:
            # Read .txt and .md files directly — no unstructured needed
            text = file_path.read_text(encoding="utf-8")
            docs = [Document(page_content=text, metadata={"source": file_path.name})]

        documents.extend(docs)

    return documents

# ── Chunk ─────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """Split documents into chunks sized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)

# ── Embed and store ───────────────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks with OpenAI and persist to ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",    # cheap, high quality
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore

# ── Test retrieval ────────────────────────────────────────────────────────────

def test_retrieval(vectorstore: Chroma) -> None:
    """Run a few test queries and print retrieved chunks so you can verify quality."""
    test_questions = [
        "How do I apply for a role?",
        "Do you offer visa sponsorship?",
        "How long does the hiring process take?",
    ]

    print("\n── Retrieval test ───────────────────────────────────────────")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    for question in test_questions:
        print(f"\nQ: {question}")
        results = retriever.invoke(question)
        for i, doc in enumerate(results, 1):
            print(f"  Chunk {i} [{doc.metadata.get('source', 'unknown')}]:")
            print(f"  {doc.page_content[:200].strip()}...")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── RAG Ingestion Pipeline ───────────────────────────────────")

    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"\nNo documents found in {DOCS_DIR}/")
        print("Add .txt, .md, or .pdf files to the docs/ folder and re-run.")
        return

    # Step 1: Load
    print(f"\n1. Loading documents from {DOCS_DIR}/")
    documents = load_documents(DOCS_DIR)
    print(f"   Loaded {len(documents)} document(s)")

    # Step 2: Chunk
    print(f"\n2. Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Embed and store
    print(f"\n3. Embedding with text-embedding-3-small and storing in ChromaDB")
    vectorstore = build_vectorstore(chunks)
    print(f"   Stored {len(chunks)} chunks in {CHROMA_DIR}/")

    # Step 4: Test
    print("\n4. Testing retrieval...")
    test_retrieval(vectorstore)

    print("\n✓ Ingestion complete. Run chat.py to test the full pipeline.\n")

if __name__ == "__main__":
    main()