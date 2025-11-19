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
        
        response_html = """
        <div class="answer-section" style="background: #d4edda; border-left: 4px solid #28a745;">
            <strong>✅ Conversation Cleared!</strong><br><br>
            Your chat history has been reset. You can start a fresh conversation now.<br><br>
            Feel free to ask me any medical questions!
        </div>
        """
        return response_html
    
    # Handle "help" command
    if question_lower in ["help", "commands", "?", "how to use"]:
        print("📖 Help command executed")
        
        response_html = """
        <div class="answer-section" style="background: #e7f3ff; border-left: 4px solid #0066cc;">
            <strong>📚 How to Use Aceso Medical AI</strong><br><br>
            
            <strong>Ask Medical Questions:</strong>
            <ul>
                <li>"What are symptoms of diabetes?"</li>
                <li>"Treatment options for hypertension?"</li>
                <li>"Did WHO release dengue advisories?"</li>
            </ul>
            
            <strong>Commands:</strong>
            <ul>
                <li>Type <code>help</code> - Show this help message</li>
                <li>Type <code>reset</code> - Clear conversation history</li>
                <li>Type <code>reasoning on/off</code> - Toggle reasoning display</li>
            </ul>
            
            <strong>Features:</strong>
            <ul>
                <li>✓ Search medical databases for established facts</li>
                <li>✓ Search web for current guidelines and advisories</li>
                <li>✓ Show reasoning process behind answers</li>
                <li>✓ Cite sources for all information</li>
            </ul>
            
            <div style="background: #fff3cd; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <strong>⚠️ Disclaimer:</strong> This is for educational purposes only. Always consult healthcare professionals for medical advice.
            </div>
        </div>
        """
        return response_html
    
    # Handle reasoning toggle
    if question_lower in ["reasoning on", "reasoning off"]:
        state = "ON" if "on" in question_lower else "OFF"
        print(f"🔧 Reasoning display: {state}")
        
        response_html = f"""
        <div class="answer-section" style="background: #fff3cd; border-left: 4px solid #ffc107;">
            <strong>✅ Reasoning Display: {state}</strong><br><br>
            The reasoning process will {'now be visible' if state == 'ON' else 'be hidden'} in subsequent responses.<br><br>
            {'You can see how I think through problems!' if state == 'ON' else 'Responses will be cleaner and faster.'}
        </div>
        """
        return response_html
    
    # ✅ Not a command - proceed with normal LLM processing
    print(f"🔍 Processing: {question}")
    
    # Get conversation history
    history = session.get("history", [])

    # Limit history size - keep only last 4 exchanges (8 messages)
    if len(history) > 8:
        history = history[-8:]
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
        "safety_score": 0.0,
        "show_reasoning": show_reasoning,
        "used_tools": [],
        "full_observations": [],
        "last_action": {},
        "forced_web_search": False
    }
    
    try:
        result = agentic_rag.invoke(
            state,
            config={"recursion_limit": 25}
        )
        answer = result.get("draft") or "⚠️ I couldn't generate a safe answer."
        
        # Separate reasoning from answer
        reasoning_html, final_answer = clean_reasoning_trace(answer)
        
        # Escape HTML to prevent breaking out of container
        import html
        final_answer_safe = html.escape(final_answer).replace('\n', '<br>')
        reasoning_html_safe = html.escape(reasoning_html).replace('\n', '<br>') if reasoning_html else ""
        
        # Save to history (truncated for session storage)
        history.append({"role": "user", "content": question[:300]})
        history.append({"role": "assistant", "content": final_answer[:600]})
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
        
        # User-friendly error messages
        error_str = str(e).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return """
            <div style="background: #fff8dc; padding: 15px; border-radius: 8px; border-left: 4px solid #ffa500;">
                <strong>⏱️ Processing Timeout</strong><br><br>
                Your question is complex and took too long to process.<br><br>
                <strong>Try these strategies:</strong>
                <ul>
                    <li>Break the question into parts</li>
                    <li>Simplify the question</li>
                    <li>Type <code>reset</code> to clear history</li>
                </ul>
            </div>
            """
        
        if "connection" in error_str:
            return """
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
            Type <code>reset</code> to clear history or try rephrasing your question.
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