from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from ddgs import DDGS
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template, chitchat_prompt
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import requests
import os, re

load_dotenv()

# ---- Vector store ----
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
web_search = DuckDuckGoSearchResults()

# ---- Prompt ----
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)

# ---- Reranker ----
reranker = CrossEncoder("BAAI/bge-reranker-large")

# ---- State ----
class RAGState(TypedDict):
    question: str
    route: Optional[Literal["chitchat","definition","guideline","general"]]
    contexts: List[Document]
    draft: Optional[str]
    grounded_score: float
    safety_score: float
    tries: int
    history: List[str]   # NEW: keep short conversation history
    best_score: float
    best_score_pine: float
    best_score_web: float
    did_web: bool


# ---- LLM (via Ollama) ----
llm = ChatOllama(
    model="llama3",   # pulled with `ollama pull`
    temperature=0.2,
    num_predict=512
)

# ---- Nodes ----
import re

def classify(state: RAGState) -> RAGState:
    q = state["question"].lower().strip()

    # Detect chit-chat by phrases
    chit_phrases = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "who are you", "thank you", "thanks"
    ]
    if any(phrase in q for phrase in chit_phrases):
        state["route"] = "chitchat"
        return state

    # Definition-type questions
    if any(k in q for k in ["what is", "define", "definition", "meaning"]):
        state["route"] = "definition"
    else:
        # Everything else is general
        state["route"] = "general"

    print(f">>> Classified as {state['route']}")
    return state



def chitchat(state: RAGState) -> RAGState:
    prompt = chitchat_prompt.format(question=state["question"])
    out = llm.invoke(prompt)

    if hasattr(out, "content"):
        state["draft"] = out.content
    else:
        state["draft"] = str(out)

    return state


def retrieve(state: RAGState) -> RAGState:
    k = 6
    retriever = docsearch.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(state["question"]) 

    # save contexts
    state["contexts"] = docs

    # quick debug
    print(f">>> Retrieved {len(docs)} docs from Pinecone")

    return state

def rerank_node(state: RAGState) -> RAGState:
    if not state["contexts"]:
        return state

    # Score all docs
    pairs = [(state["question"], d.page_content) for d in state["contexts"]]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(state["contexts"], scores), key=lambda x: x[1], reverse=True)

    # Split by source
    pine_docs = [(d, s) for d, s in ranked if d.metadata.get("source") != "web"]
    web_docs  = [(d, s) for d, s in ranked if d.metadata.get("source") == "web"]

    # Take top-2 Pinecone + top-2 Web (if available)
    top_pine = [d for d, _ in pine_docs[:2]]
    top_web  = [d for d, _ in web_docs[:2]]

    # Merge sets (prioritize balance)
    top_docs = top_pine + top_web

    # Fallback: fill up to 4 with highest remaining docs
    if len(top_docs) < 4:
        extra = [d for d, _ in ranked if d not in top_docs]
        top_docs.extend(extra[: 4 - len(top_docs)])

    # Ensure at least 1 doc survives
    state["contexts"] = top_docs or [state["contexts"][0]]

    # Compute scores separately
    pine_scores = [s for d, s in ranked if d in top_pine]
    web_scores  = [s for d, s in ranked if d in top_web]

    state["best_score_pine"] = max(pine_scores) if pine_scores else 0.0
    state["best_score_web"]  = max(web_scores) if web_scores else 0.0
    state["best_score"]      = max(state["best_score_pine"], state["best_score_web"])

    print(
        f">>> Rerank best_score={state['best_score']:.3f} "
        f"(pine={state['best_score_pine']:.3f}, web={state['best_score_web']:.3f})"
    )

    return state


def response(state: RAGState) -> RAGState:
    ctx = "\n\n".join([d.page_content for d in state["contexts"]])
    history_text = "\n".join(state.get("history", []))

    out = llm.invoke(PROMPT.format(
        history=history_text,
        question=state["question"],
        context=ctx
    ))
    text = out.content if hasattr(out, "content") else str(out)
    text = text.strip()

    # Collect refs
    web_refs, book_refs = [], []
    for d in state["contexts"]:
        source = d.metadata.get("source", "")
        if source == "web":
            url = d.metadata.get("url")
            title = d.metadata.get("title", "Web Source")
            if url:
                web_refs.append(f"🌐 {title}: {url}")
            else:
                web_refs.append("🌐 Web Search Result")
        elif "Gale Encyclopedia" in source or source == "book":
            page = d.metadata.get("page", "?")
            para = d.metadata.get("paragraph", "?")
            book_refs.append(f"📖 Gale Encyclopedia (page {page}, paragraph {para})")

    # ✅ Choose what to display
    if state.get("did_web", False) and web_refs:
        refs = web_refs   # prefer web refs if web_search was used
    elif book_refs:
        refs = book_refs
    else:
        refs = []

    final_answer = (
        f"{text}\n\n"
        "⚠️ Disclaimer: This information is for educational purposes only. "
        "Please consult a qualified healthcare professional for personal medical advice.\n\n"
    )
    if refs:
        final_answer += "References:\n" + "\n".join(refs)

    state["draft"] = final_answer
    return state

def grade(state: RAGState) -> RAGState:
    text = state.get("draft") or ""
    has_refs = "References:" in text or "📖" in text or "🌐" in text
    grounded = 0.9 if has_refs else 0.5

    # simple safety screen
    unsafe = re.search(r"\b(diagnose|guaranteed cure|definitive)\b", text.lower())
    safety = 0.9 if not unsafe else 0.3

    # debug
    print(">>> Grading: has_refs=", has_refs, "grounded=", grounded)

    state["grounded_score"] = grounded
    state["safety_score"] = safety
    return state

def refine_or_finish(state: RAGState) -> RAGState:
    g = state.get("grounded_score", 0.0)
    s = state.get("safety_score", 0.0)

    # Safety first
    if s < 0.6:
        state["draft"] = (
            "⚠️ I cannot provide a safe or definitive medical answer based on the available context. "
            "Please consult a qualified healthcare professional."
        )
        return state

    # Hint the next pass only on the very first loop if poorly grounded
    if state.get("tries", 0) == 0 and g < 0.7:
        state["question"] += " (rephrased for medical terminology)"
        print(">>> Rephrasing hint added for next pass")

    # Do NOT change 'tries' here — counting happens only in loop_or_end
    return state


def web_search_tool(state: RAGState) -> RAGState:
    query = state["question"]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if results:
            # Append web results (keep Pinecone + Web together)
            new_web_docs = [
                Document(
                    page_content=r.get("body") or r.get("snippet") or "",
                    metadata={
                        "source": "web",
                        "url": r.get("href", ""),
                        "title": r.get("title", "Web Result"),
                    },
                )
                for r in results[:3]
            ]
            state["contexts"].extend(new_web_docs)
        else:
            state["contexts"].append(
                Document(
                    page_content="(no web results found)",
                    metadata={"source": "web"},
                )
            )
    except Exception as e:
        state["contexts"].append(
            Document(
                page_content=f"(web search failed: {str(e)})",
                metadata={"source": "web"},
            )
        )

    # Flag to avoid infinite web retries
    state["did_web"] = True  

    print(">>> Entered web_search_tool")
    print(
        f"Now total contexts = {len(state['contexts'])} "
        f"({sum(1 for d in state['contexts'] if d.metadata.get('source') != 'web')} Pinecone + "
        f"{sum(1 for d in state['contexts'] if d.metadata.get('source') == 'web')} Web)"
    )
    return state





# ---- Build Graph ----
# Nodes
graph = StateGraph(RAGState)
graph.add_node("classify", classify)    
graph.add_node("chitchat", chitchat)
graph.add_node("web_search", web_search_tool)
graph.add_node("retrieve", retrieve)
graph.add_node("rerank", rerank_node)
graph.add_node("response", response)
graph.add_node("grade", grade)
graph.add_node("refine_or_finish", refine_or_finish)

# Edges
graph.set_entry_point("classify")
# Conditional routing based on classify
graph.add_conditional_edges(
    "classify",
    lambda s: s["route"],
    {
        "chitchat": "chitchat", # greetings + chit-chat
        "definition": "retrieve",
        "general": "retrieve"
    }
)

# Main pipeline
graph.add_edge("retrieve", "rerank")
graph.add_edge("web_search", "rerank")
graph.add_edge("rerank", "response")
graph.add_edge("response", "grade")
graph.add_edge("grade", "refine_or_finish")

def loop_or_end(state: RAGState):
    # Increment tries ONCE per loop here
    state["tries"] = state.get("tries", 0) + 1
    tries = state["tries"]
    grounded = state.get("grounded_score", 0.0)
    best_score = state.get("best_score", 0.0)
    did_web = state.get("did_web", False)
    route = state.get("route", "general")

    print(f"[loop_or_end] tries={tries}, grounded={grounded}, best_score={best_score}, did_web={did_web}")

    # Hard stop after 3 attempts total
    if tries >= 3:
        print(">>> Max retries reached → END")
        return END
    
    # If definition → never go to web
    if route == "definition":
        if tries == 1 and state.get("best_score_pine", 0.0) < 0.5 and grounded < 0.8:
            print(">>> Definition route → retrying Pinecone only")
            return "retrieve"
        print(">>> Definition route → stopping (no web search allowed)")
        return END

    # If Pinecone looks weak, do web search exactly once
    if state.get("best_score_pine", 0.0) < 0.5 and not did_web:
        print(">>> Weak Pinecone docs → using web_search")
        return "web_search"

    # Retry flow based on grounding (only if still under tries limit)
    if tries == 1 and grounded < 0.8:
        return "retrieve"
    if tries == 2 and grounded < 0.8 and not did_web:
        return "web_search"

    # Otherwise we're done
    return END




graph.add_conditional_edges("refine_or_finish", loop_or_end)

agentic_rag = graph.compile()

