"""
Flask App for Medical Chatbot with ReAct Architecture
Updated to use react_agentic_app.py with conversation history
"""

from flask import Flask, render_template, request, session, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
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
SUMMARIZATION_MODEL = "llama3"
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
    llm = ChatOllama(model=SUMMARIZATION_MODEL, temperature=0.3, num_predict=256)
    
    summary_prompt = f"""Summarize this conversation history concisely, preserving key medical topics discussed:

{old_text}

Provide a brief summary (2-3 sentences) that captures the main health topics and questions."""
    
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
    
    # ... (existing command handling code) ...
    
    # Get conversation history
    history = session.get("history", [])

    # ✅ FIX 1: LIMIT HISTORY SIZE - keep only last 4 exchanges (8 messages)
    if len(history) > 8:
        history = history[-8:]  # Keep only last 8 messages
        session["history"] = history
        print(f"📝 Trimmed history to last 8 messages")

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
        "grounded_score": 0.0,
        "safety_score": 0.0,
        "show_reasoning": show_reasoning,
        "used_tools": [],
        "full_observations": [],
        "last_action": {},
        "forced_web_search": False  # ✅ ADD THIS
    }

    print(f"🔍 Processing: {question}")
    
    try:
        result = agentic_rag.invoke(
            state,
            config={"recursion_limit": 25}
        )
        answer = result.get("draft") or "⚠️ I couldn't generate a safe answer."
        
        # Separate reasoning from answer
        reasoning_html, final_answer = clean_reasoning_trace(answer)
        
        # ✅ CRITICAL: Escape HTML to prevent breaking out of container
        import html
        
        # Escape the content but preserve intentional line breaks
        final_answer_safe = html.escape(final_answer).replace('\n', '<br>')
        reasoning_html_safe = html.escape(reasoning_html).replace('\n', '<br>') if reasoning_html else ""
        
        # Save to history
        # ✅ FIX 2: TRUNCATE LONG CONTENT for session storage
        history.append({"role": "user", "content": question[:300]})  # Max 300 chars
        history.append({"role": "assistant", "content": final_answer[:600]})  # Max 600 chars
        session["history"] = history
        
        # Build response
        response_html = ""

        if reasoning_html_safe and show_reasoning:
            response_html += f"""
            <div class="reasoning-section">
                <details open>
                    <summary style="cursor: pointer; font-weight: bold; color: #0066cc; margin-bottom: 10px;">
                        🔍 Reasoning Process
                    </summary>
                    <pre style="margin: 0; padding: 10px; background: #ffffff; border-radius: 4px; overflow-x: auto;">{reasoning_html_safe}</pre>
                </details>
            </div>
            """     

        response_html += f"""
        <div class="answer-section">
            {final_answer_safe}
        </div>
        """

        
        print(f"✅ Response generated: {len(response_html)} chars")
        return response_html
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # ✅ User-friendly error messages
        error_str = str(e).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return f"""
            <div style="background: #fff8dc; padding: 15px; border-radius: 8px; border-left: 4px solid #ffa500;">
                <strong>⏱️ Processing Timeout</strong><br><br>
                Your question is complex and took too long to process.<br><br>
                <strong>Try these strategies:</strong>
                <ul>
                    <li>Break the question into parts (ask about treatment first, then new drugs separately)</li>
                    <li>Simplify: "What are common Parkinson's treatments?"</li>
                    <li>Disable reasoning traces (click the toggle button)</li>
                </ul>
            </div>
            """
        
        if "connection" in error_str:
            return f"""
            <div style="background: #ffe6e6; padding: 15px; border-radius: 8px; border-left: 4px solid #ff0000;">
                <strong>🔌 Connection Error</strong><br><br>
                Cannot connect to Ollama server.<br><br>
                <strong>Check:</strong>
                <ul>
                    <li>Is Ollama running? (<code>ollama serve</code>)</li>
                    <li>Is the model downloaded? (<code>ollama pull llama3</code>)</li>
                </ul>
            </div>
            """
        
        return f"""
        <div style="background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545;">
            <strong>⚠️ Error Occurred</strong><br><br>
            {str(e)[:200]}<br><br>
            Please try rephrasing your question or type <code>reset</code> to clear history.
        </div>
        """


@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history (for debugging or UI display)."""
    history = session.get("history", [])
    return jsonify({
        "history": history,
        "turn_count": len(history) // 2,
        "reasoning_enabled": ENABLE_REASONING_TRACES
    })


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history."""
    session["history"] = []
    return jsonify({"message": "History cleared"})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "system": "ReAct Medical Chatbot",
        "version": "2.0"
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
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("FLASK_DEBUG", "True").lower() == "true"
    )