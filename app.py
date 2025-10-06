from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from src.graph_app import agentic_rag
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session
load_dotenv()
app.has_reset = False 

## RAG Code (Old)

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# embeddings = download_hugging_face_embeddings()

# index_name="medical-chatbot"

# docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# chain_type_kwargs={"prompt": PROMPT}

# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})


# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     result=qa({"query": input})
#     print("Response : ", result["result"])
#     return str(result["result"])

def summarize_history(full_history, llm, max_turns=10):
    """Summarize older history if too long, keep last N turns intact."""
    if len(full_history) <= max_turns * 2:  # 2 entries per turn (User + Bot)
        return "\n".join(full_history), None  # return as plain string

    # Split into old and recent parts
    old_part = "\n".join(full_history[:-max_turns*2])
    recent_part = "\n".join(full_history[-max_turns*2:])

    # Summarize old part
    prompt = f"""
    Summarize the following conversation history in a concise way,
    keeping key user questions and assistant responses:

    {old_part}

    Return a short summary that preserves medical context.
    """
    summary_out = llm.invoke(prompt)
    summary = summary_out.content if hasattr(summary_out, "content") else str(summary_out)

    # Build final summarized history string
    summarized_history = f"Summary so far: {summary}\n\nRecent conversation:\n{recent_part}"
    return summarized_history, summary


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    question = request.form["msg"].strip()

    if question.lower() == "reset":
        session["history"] = []
        return "🧹 History cleared. Let's start fresh!"

    # Retrieve full history
    history = session.get("history", [])

    # 🔑 Use LLaMA-3 via Ollama for summarization
    llm_for_history = ChatOllama(
        model="llama3:8b",      # or llama3:70b if you’ve got GPU/VRAM
        temperature=0.3,
        max_tokens=256
    )

    # 🔑 Summarize if long
    summarized_history, summary = summarize_history(history, llm_for_history)

    state = {
        "question": question,
        "route": None,
        "contexts": [],
        "draft": None,
        "citations": [],
        "grounded_score": 0.0,
        "safety_score": 0.0,
        "tries": 0,
        "history": summarized_history,  # send to agent
        "did_web": False,
    }

    result = agentic_rag.invoke(state)
    answer = result.get("draft") or "Sorry, I couldn’t find a safe, grounded answer."

    # Save turn
    history.append(f"User: {question}")
    history.append(f"Bot: {answer}")
    session["history"] = history  

    return answer



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)