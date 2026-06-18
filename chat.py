"""
chat.py — RAG Recruitment Chatbot
Loads the ChromaDB vector store, retrieves relevant chunks,
and uses GPT-4o-mini to answer candidate questions.

Run from the terminal to test the full pipeline:
    python chat.py

Make sure you have run ingest.py first to populate chroma_db/.
"""

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_DIR  = "chroma_db"
COLLECTION  = "recruitment"
NUM_CHUNKS  = 4     # number of chunks to retrieve per question

# ── Load vectorstore ──────────────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    """Load the existing ChromaDB vector store from disk."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore

# ── Build RAG chain ───────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: Chroma):
    """Build the full retrieval + LLM chain."""

    # Retriever — pulls top NUM_CHUNKS most relevant chunks for each question
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": NUM_CHUNKS}
    )

    # LLM — GPT-4o-mini is fast and cheap, well suited for FAQ responses
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,                      # deterministic answers
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a helpful recruitment assistant for MIP's Analytics Career Accelerator program.
Answer the candidate's question using only the information provided in the context below.

Rules:
- Be friendly, clear, and concise — 2 to 4 sentences unless the question genuinely needs more
- If the answer is not in the context, say you don't have that information and suggest the candidate emails the team at careers@mipaustralia.com.au
- Never make up details that are not in the context
- If asked about salary, visas, or program dates, be precise — do not speculate

Context:
{context}

Question: {input}

Answer:""")

    # Chain — stuffs retrieved chunks into the prompt then calls the LLM
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    return rag_chain

# ── Debug helper ──────────────────────────────────────────────────────────────

def show_retrieved_chunks(vectorstore: Chroma, question: str) -> None:
    """Print the chunks retrieved for a question — useful for debugging."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})
    chunks = retriever.invoke(question)
    print(f"\n  [DEBUG] Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        preview = chunk.page_content[:120].replace("\n", " ").strip()
        print(f"  {i}. [{source}] {preview}...")

# ── Terminal chat loop ────────────────────────────────────────────────────────

def run_chat(rag_chain, vectorstore: Chroma, debug: bool = False) -> None:
    """Run an interactive chat loop in the terminal."""
    print("\n── MIP Recruitment Chatbot ──────────────────────────────────")
    print("Ask a question about the Analytics Career Accelerator program.")
    print("Type 'debug' to toggle chunk visibility. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question.lower() == "debug":
            debug = not debug
            print(f"  [Debug mode {'ON' if debug else 'OFF'}]")
            continue

        if debug:
            show_retrieved_chunks(vectorstore, question)

        # Get answer
        response = rag_chain.invoke({"input": question})
        print(f"\nAssistant: {response['answer']}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── Loading RAG pipeline... ──────────────────────────────────")

    # Check chroma_db exists
    if not os.path.exists(CHROMA_DIR):
        print("\nNo chroma_db/ folder found.")
        print("Run ingest.py first to build the vector database.")
        return

    # Load vectorstore
    print("Loading vector store...")
    vectorstore = load_vectorstore()

    # Build chain
    print("Building RAG chain...")
    rag_chain = build_rag_chain(vectorstore)

    print("Ready.\n")

    # Run interactive chat
    run_chat(rag_chain, vectorstore, debug=False)

if __name__ == "__main__":
    main()