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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "recruitment"
NUM_CHUNKS = 4

# ── Load vectorstore ──────────────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

# ── Build RAG chain ───────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful recruitment assistant for MIP's Analytics Career Accelerator program.
Answer the candidate's question using only the information provided in the context below.

Rules:
- Be friendly, clear, and concise - 2 to 4 sentences unless the question genuinely needs more
- If the answer is not in the context, say you don't have that information and suggest the candidate emails the team at careers@mipaustralia.com.au
- Never make up details that are not in the context
- If asked about salary, visas, or program dates, be precise - do not speculate
- If the context does not clearly answer the question, say "I don't have enough information to answer that accurately" and direct them to careers@mipaustralia.com.au — do not guess                                           

Context:
{context}

Question: {input}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── Debug helper ──────────────────────────────────────────────────────────────

def show_retrieved_chunks(retriever, question: str) -> None:
    chunks = retriever.invoke(question)
    print(f"\n  [DEBUG] Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        preview = chunk.page_content[:120].replace("\n", " ").strip()
        print(f"  {i}. [{source}] {preview}...")

# ── Terminal chat loop ────────────────────────────────────────────────────────

def run_chat(chain, retriever, debug: bool = False) -> None:
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
            show_retrieved_chunks(retriever, question)

        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── Loading RAG pipeline... ──────────────────────────────────")

    if not os.path.exists(CHROMA_DIR):
        print("\nNo chroma_db/ folder found.")
        print("Run ingest.py first to build the vector database.")
        return

    print("Loading vector store...")
    vectorstore = load_vectorstore()

    print("Building RAG chain...")
    chain, retriever = build_rag_chain(vectorstore)

    print("Ready.")

    run_chat(chain, retriever, debug=False)

if __name__ == "__main__":
    main()