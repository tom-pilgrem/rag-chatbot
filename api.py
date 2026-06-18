"""
api.py — RAG Recruitment Chatbot API
FastAPI backend that wraps the RAG chain and serves it over HTTP.

Run locally with:
    uvicorn api:app --reload

Then open http://localhost:8000 in your browser to see the chat widget.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "recruitment"
NUM_CHUNKS = 4

# ── Load RAG chain (once at startup) ─────────────────────────────────────────

def load_chain():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful recruitment assistant for MIP's Analytics Career Accelerator program.
Answer the candidate's question using only the information provided in the context below.

Rules:
- Be friendly, clear, and concise - 2 to 4 sentences unless the question genuinely needs more
- If the answer is not in the context, say you don't have that information and suggest the candidate emails the team at careers@mipaustralia.com.au
- Never make up details that are not in the context
- If asked about salary, visas, or program dates, be precise - do not speculate
- If the context does not clearly answer the question, say you don't have enough information and direct them to careers@mipaustralia.com.au - do not guess

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_with_history(x):
        question = x["input"]
        history = x["chat_history"]

        if not history:
            return retriever.invoke(question)

        history_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history[-4:]
        ])
        rewrite_prompt = f"""Given this conversation:
{history_text}

Rewrite this follow-up as a standalone question: "{question}"
Return only the rewritten question, nothing else."""

        rewritten = llm.invoke(rewrite_prompt).content
        return retriever.invoke(rewritten)

    def get_context(x):
        docs = retrieve_with_history(x)
        return format_docs(docs)

    chain = (
        {
            "context": get_context,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain before going live
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading RAG chain...")
chain = load_chain()
print("Ready.")

sessions: dict[str, list] = {}

# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    session_id: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = sessions.get(request.session_id, [])

    lc_history = []
    for msg in history:
        if msg["role"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    answer = chain.invoke({
        "input": request.message,
        "chat_history": lc_history,
    })

    history.append({"role": "human", "content": request.message})
    history.append({"role": "ai", "content": answer})
    sessions[request.session_id] = history

    return ChatResponse(answer=answer, session_id=request.session_id)


@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_widget():
    return HTMLResponse(content=open("widget.html").read())