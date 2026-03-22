"""
Flask App for Medical Chatbot with ReAct Architecture
Updated to use react_agentic_app.py with conversation history
"""

from flask import Flask, render_template, request, session, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.timeout = 180  # 90 seconds

# Import the NEW ReAct system
from src.react_agentic_app import agentic_rag

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # Use env var for security

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_HISTORY_TURNS = 10  # Keep last 10 conversation turns
SUMMARIZATION_MODEL = "llama-3.1-8b-instant"
ENABLE_REASONING_TRACES = True  # Global default for showing reasoning


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def summarize_history(full_history, max_turns=MAX_HISTORY_TURNS):
    """
    Summarize conversation history if too long.
    Keeps recent turns intact, summarizes older context.
    
    Args:
        full_history: List of message dicts [{"role": "user", "content": "..."}, ...]
        max_turns: Number of recent turns to keep unsummarized
        
    Returns:
        List of messages (LangChain format) with summarized history
    """
    if len(full_history) <= max_turns * 2:  # 2 messages per turn
        # Convert to LangChain message format
        return [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in full_history
        ]
    
    # Split into old and recent
    old_messages = full_history[: -(max_turns * 2)]
    recent_messages = full_history[-(max_turns * 2) :]
    
    # Create text representation for summarization
    old_text = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in old_messages
    ])
    
    # Summarize old part
    llm = ChatGroq(model=SUMMARIZATION_MODEL, temperature=0.3, groq_api_key=os.getenv("GROQ_API_KEY"))
    
    summary_prompt = f"""Summarize this conversation history concisely, preserving key medical topics discussed:

{old_text}

Provide a brief summary that captures the main health topics and questions."""
    
    try:
        summary_response = llm.invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
        
        # Create summary message
        summary_msg = AIMessage(content=f"[Conversation Summary] {summary}")
        
        # Convert recent messages to LangChain format
        recent_msgs = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in recent_messages
        ]
        
        return [summary_msg] + recent_msgs
        
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback: just convert recent messages
        return [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in recent_messages
        ]


def clean_reasoning_trace(answer: str) -> tuple[str, str]:
    """
    Separate reasoning trace from final answer.
    
    Returns:
        (reasoning_html, final_answer)
    """
    if "## 🔍 Reasoning Process" not in answer:
        return "", answer
    
    parts = answer.split("## 📋 Final Answer")
    
    if len(parts) == 2:
        reasoning = parts[0].replace("## 🔍 Reasoning Process", "").strip()
        final_answer = parts[1].strip()
        return reasoning, final_answer
    
    return "", answer


# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Render main chat interface."""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Main chat endpoint - processes user messages through ReAct system.
    """
    question = request.form.get("msg", "").strip()
    
    if not question:
        return jsonify({"error": "Empty message"}), 400
    
    # ✅ COMMAND INTERCEPTION - Handle special commands BEFORE LLM processing
    question_lower = question.lower()
    
    # Handle "reset" / "clear" command
    if question_lower in ["reset", "clear", "restart", "new"]:
        session["history"] = []
        print("🔄 History cleared via command")
        
        return jsonify({
            "type": "command",
            "command": "reset",
            "message": "✅ Conversation Cleared!",
            "details": "Your chat history has been reset. You can start a fresh conversation now. Feel free to ask me any medical questions!"
        })
    
    # Handle "help" command
    if question_lower in ["help", "commands", "?", "how to use"]:
        print("📖 Help command executed")
        
        return jsonify({
            "type": "command",
            "command": "help",
            "message": "📚 How to Use Aceso Medical AI",
            "sections": [
                {
                    "title": "Ask Medical Questions",
                    "items": [
                        "What are symptoms of diabetes?",
                        "Treatment options for hypertension?",
                        "Did WHO release dengue advisories?"
                    ]
                },
                {
                    "title": "Commands",
                    "items": [
                        "Type 'help' - Show this help message",
                        "Type 'reset' - Clear conversation history",
                        "Type 'reasoning on/off' - Toggle reasoning display"
                    ]
                },
                {
                    "title": "Features",
                    "items": [
                        "✓ Search medical databases for established facts",
                        "✓ Search web for current guidelines and advisories",
                        "✓ Show reasoning process behind answers",
                        "✓ Cite sources for all information"
                    ]
                }
            ],
            "disclaimer": "⚠️ Disclaimer: This is for educational purposes only. Always consult healthcare professionals for medical advice."
        })
    
    # Handle reasoning toggle
    if question_lower in ["reasoning on", "reasoning off"]:
        state = "ON" if "on" in question_lower else "OFF"
        print(f"🔧 Reasoning display: {state}")
        
        return jsonify({
            "type": "command",
            "command": "reasoning_toggle",
            "message": f"✅ Reasoning Display: {state}",
            "details": f"The reasoning process will {'now be visible' if state == 'ON' else 'be hidden'} in subsequent responses. {'You can see how I think through problems!' if state == 'ON' else 'Responses will be cleaner and faster.'}"
        })
    
    # ✅ Not a command - proceed with normal LLM processing
    print(f"🔍 Processing: {question}")
    
    # Get conversation history
    history = session.get("history", [])

    # Let summarize_history handle the logic
    # If > 20 messages (10 turns), it will summarize old ones and keep recent 10 turns
    message_history = summarize_history(history)
    message_history.append(HumanMessage(content=question))
    
    show_reasoning = request.form.get("show_reasoning", str(ENABLE_REASONING_TRACES)).lower() == "true"
    
    state = {
        "messages": message_history,
        "question": question,
        "reasoning_trace": [],
        "step_count": 0,
        "contexts": [],
        "draft": "",
        "safety_score": 0.0,
        "show_reasoning": show_reasoning,
        "used_tools": [],
        "full_observations": [],
        "last_action": {},
        "forced_web_search": False,
        "tool_history": []
    }
    
    try:
        result = agentic_rag.invoke(
            state,
            config={"recursion_limit": 25}
        )
        answer = result.get("draft") or "⚠️ I couldn't generate a safe answer."
        
        # Separate reasoning from answer
        reasoning_text, final_answer = clean_reasoning_trace(answer)
        
        # Parse sources from answer AND strip them from the answer body
        sources = []
        answer_body = final_answer
        if "## 📚 Sources" in final_answer:
            parts = final_answer.split("## 📚 Sources")
            # Keep only the answer body (before sources section)
            answer_body = parts[0].rstrip().rstrip("---").strip()
            if len(parts) > 1:
                source_lines = parts[1].strip().split('\n')
                for line in source_lines:
                    line = line.strip()
                    if line and (line.startswith('📖') or line.startswith('🌐')):
                        sources.append(line)

        # Fixed disclaimer — always shown once at the bottom by the frontend
        disclaimer = "⚠️ Disclaimer: This information is for educational purposes only. Please consult a qualified healthcare professional for personal medical advice."

        # Save to history (truncated for session storage)
        history.append({"role": "user", "content": question[:300]})
        history.append({"role": "assistant", "content": answer_body[:600]})

        # Trim session storage to prevent excessive bloat (keep last 40 messages = 20 turns)
        # This allows summarize_history to trigger when history exceeds 20 messages
        # But prevents unlimited session growth beyond 40 messages
        if len(history) > 40:
            history = history[-40:]
            print(f"📝 Trimmed session storage to last 40 messages")

        session["history"] = history

        # Return clean JSON data (no HTML)
        response_data = {
            "answer": answer_body.strip(),
            "reasoning": reasoning_text.strip() if reasoning_text else "",
            "sources": sources,
            "disclaimer": disclaimer,
            "show_reasoning": show_reasoning,
            "type": "answer"
        }
        
        print(f"✅ Response generated: {len(final_answer)} chars")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # User-friendly error messages
        error_str = str(e).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return jsonify({
                "type": "error",
                "error_type": "timeout",
                "message": "⏱️ Processing Timeout",
                "details": "Your question is complex and took too long to process.",
                "suggestions": [
                    "Break the question into parts",
                    "Simplify the question",
                    "Type 'reset' to clear history"
                ]
            })
        
        if "connection" in error_str:
            return jsonify({
                "type": "error",
                "error_type": "connection",
                "message": "🔌 Connection Error",
                "details": "Cannot connect to Ollama server.",
                "suggestions": [
                    "Is GROQ_API_KEY set in your environment?",
                    "Check your Groq API key at console.groq.com"
                ]
            })
        
        return jsonify({
            "type": "error",
            "error_type": "general",
            "message": "⚠️ Error Occurred",
            "details": str(e)[:200],
            "suggestions": [
                "Type 'reset' to clear history or try rephrasing your question."
            ]
        })


@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history (for debugging or UI display)."""
    history = session.get("history", [])
    return jsonify({
        "history": history,
        "turn_count": len(history) // 2,
        "reasoning_enabled": ENABLE_REASONING_TRACES
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 Medical Chatbot with ReAct Architecture")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   - Max history turns: {MAX_HISTORY_TURNS}")
    print(f"   - Summarization model: {SUMMARIZATION_MODEL}")
    print(f"   - Reasoning traces: {'Enabled' if ENABLE_REASONING_TRACES else 'Disabled'}")
    print("=" * 70)
    print("🚀 Starting Flask server...")
    print("=" * 70)
    
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        debug=os.getenv("FLASK_DEBUG", "True").lower() == "true"
    )