from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
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


@app.before_request
def clear_session_once():
    if not app.has_reset:
        session.clear()
        app.has_reset = True
        print(">>> Session history cleared on app startup")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    question = request.form["msg"].strip()

    # 🧹 Reset history if user sends "reset"
    if question.lower() == "reset":
        session["history"] = []
        return "🧹 History cleared. Let's start fresh!"

    # Get history if it exists, else empty list
    history = session.get("history", [])

    state = {
        "question": question,
        "route": None,
        "contexts": [],
        "draft": None,
        "citations": [],
        "grounded_score": 0.0,
        "safety_score": 0.0,
        "tries": 0,
        "history": history[-4:],  # keep last 4 turns
        "did_web": False, 
    }

    result = agentic_rag.invoke(state)
    answer = result.get("draft") or "Sorry, I couldn’t find a safe, grounded answer."

    # Save this turn into session history
    history.append(f"User: {question}")
    history.append(f"Bot: {answer}")
    session["history"] = history  # save back into session

    return answer


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)