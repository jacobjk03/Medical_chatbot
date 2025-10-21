"""
Medical Chatbot with True ReAct Architecture
Shows explicit Thought → Action → Observation traces to user
"""

from typing import TypedDict, List, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from sentence_transformers import CrossEncoder
from ddgs import DDGS
from dotenv import load_dotenv
import os
import re

from src.helper import download_hugging_face_embeddings
from src.prompt import (
    react_system_prompt,
    safety_prompt,
    polish_prompt,
    synthesis_prompt
)

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

PINECONE_INDEX = "medical-chatbot"
RERANKER_MODEL = "BAAI/bge-reranker-large"
MAX_REASONING_STEPS = 5

# ============================================================================
# INITIALIZATION
# ============================================================================

embeddings = download_hugging_face_embeddings()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
docsearch = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
reranker = CrossEncoder(RERANKER_MODEL)

print("✅ ReAct Agentic RAG system initialized")

# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(TypedDict):
    """State with ReAct traces."""
    messages: Annotated[List[AnyMessage], "add_messages"]
    question: str
    reasoning_trace: List[str]
    step_count: int
    contexts: List[Document]
    draft: str
    grounded_score: float
    safety_score: float
    show_reasoning: bool
    used_tools: List[str]
    full_observations: List[str]
    last_action: dict
    forced_web_search: bool  # ✅ ADD THIS


# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def search_medical_database(query: str) -> str:
    """
    Search the medical encyclopedia database for reliable, established medical information.
    Use this for:
    - Medical definitions and terminology
    - Established medical facts
    - General health information
    - Symptoms and conditions
    """
    retriever = docsearch.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(query)
    
    # Rerank
    if docs:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:4]]
    else:
        top_docs = docs[:4] if docs else []
    
    # Format results
    results = []
    for doc in top_docs:
        source = doc.metadata.get("source", "Medical Encyclopedia")
        page = doc.metadata.get("page", "")
        content_preview = doc.page_content[:1000] + "..."
        results.append(f"📖 {source} {f'(page {page})' if page else ''}\n{content_preview}")
    
    return "\n\n".join(results) if results else "No database results found."


@tool
def search_web_medical(query: str) -> str:
    """
    Search the web for current medical information, recent research, and latest treatments.
    Use this for:
    - Recent medical breakthroughs (2024-2025)
    - Latest treatment options
    - Current medical guidelines
    - Recent research or news
    - Questions mentioning 'latest', 'recent', 'current', or specific years
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        
        if not results:
            return "No web results found."
        
        # Format results
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            snippet = (r.get("body") or r.get("snippet") or "")[:800]
            formatted.append(f"🌐 {title}\n{url}\n{snippet}...")
        
        return "\n\n".join(formatted)
    
    except Exception as e:
        return f"Web search error: {str(e)}"


tools = [search_medical_database, search_web_medical]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm(model: str = "llama3.2:1b", temperature: float = 0.2):
    """Get LLM instance with optimized settings."""
    return ChatOllama(
        model=model, 
        temperature=temperature, 
        num_predict=250,  # ✅ Increased from 200
        num_ctx=4096,     # ✅ Increased from 2048 - can see more history
        timeout=45.0      # ✅ Increased from 30
    )

def log(node: str, message: str):
    """Consistent logging."""
    print(f"[{node.upper()}] {message}")


def parse_react_output(text: str) -> dict:
    """
    Parse ReAct format output - handles both plain and markdown formatting.
    Returns: {"thought": str, "action": str, "action_input": str}
    """
    # Try markdown format first (e.g., **Thought:** or **Action:**)
    thought_match = re.search(r'\*\*Thought[:\s]*\*\*\s*(.+?)(?=\n\*\*Action|\nAction:|$)', text, re.DOTALL | re.IGNORECASE)
    if not thought_match:
        # Try plain format
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', text, re.DOTALL | re.IGNORECASE)
    
    # Try markdown format for Action
    action_match = re.search(r'\*\*Action[:\s]*\*\*\s*(\w+)', text, re.IGNORECASE)
    if not action_match:
        # Try plain format
        action_match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
    
    # Try markdown format for Action Input
    input_match = re.search(r'\*\*Action Input[:\s]*\*\*\s*(.+?)(?=\n|$)', text, re.DOTALL | re.IGNORECASE)
    if not input_match:
        # Try plain format
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', text, re.DOTALL | re.IGNORECASE)
    
    return {
        "thought": thought_match.group(1).strip() if thought_match else "",
        "action": action_match.group(1).strip() if action_match else "",
        "action_input": input_match.group(1).strip() if input_match else ""
    }

def needs_current_info(question: str) -> bool:
    """
    Use LLM to determine if question requires current/recent information.
    Fast classification using small model.
    """
    # Cache to avoid repeated checks for same question
    if not hasattr(needs_current_info, 'cache'):
        needs_current_info.cache = {}
    
    if question in needs_current_info.cache:
        return needs_current_info.cache[question]
    
    classifier_llm = get_llm("llama3.2:1b", temperature=0.0)
    
    prompt = f"""Does this question require CURRENT or RECENT information (news, updates, outbreaks, advisories from 2024-2025)?

Question: {question}

Answer ONLY 'YES' or 'NO':
- YES if it mentions: dates, years, "latest", "recent", "current", "new", "this month/year", outbreaks, advisories, updates, "nowadays"
- NO if it asks for general medical facts, definitions, or established knowledge

Answer:"""
    
    try:
        response = classifier_llm.invoke(prompt)
        answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        result = "YES" in answer
        needs_current_info.cache[question] = result
        
        log("router", f"Temporal check: {question[:50]}... → {'CURRENT' if result else 'GENERAL'}")
        return result
    
    except Exception as e:
        log("router", f"Temporal check failed: {e}, defaulting to keyword fallback")
        # Fallback to expanded keyword list
        return keyword_temporal_check(question)


def keyword_temporal_check(question: str) -> bool:
    """
    Fallback keyword-based temporal detection.
    More comprehensive than before.
    """
    q_lower = question.lower()
    
    # Temporal indicators
    temporal_words = [
        # Time references
        "latest", "recent", "current", "new", "updated", "nowadays", "these days",
        "this year", "this month", "this week", "today", "yesterday", "last week",
        "right now", "currently", "presently",
        
        # Years (flexible)
        "2024", "2025", "2026",
        
        # Events/updates
        "outbreak", "advisory", "warning", "alert", "guideline", "recommendation",
        "release", "released", "announced", "published",
        
        # Actions requiring current data
        "spreading", "trend", "rising", "increasing"
    ]
    
    # Date patterns (e.g., "in 2025", "October 2024")
    has_date = bool(re.search(r'\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', q_lower))
    
    return any(word in q_lower for word in temporal_words) or has_date


# ============================================================================
# NODE: ReAct Reasoning
# ============================================================================

def react_reasoning(state: State) -> State:
    """
    Main ReAct reasoning node.
    LLM explicitly writes Thought → Action → Action Input
    """
    
    messages = state["messages"]
    step = state.get("step_count", 0)
    
    # Safety check
    if step >= MAX_REASONING_STEPS:
        log("react", f"⚠️ Max steps ({MAX_REASONING_STEPS}) reached - forcing finish")
    
        # Set draft to trigger generation
        state["draft"] = "MAX_STEPS_REACHED"
    
        # Set last_action to Finish so router knows to generate
        state["last_action"] = {
            "action": "Finish",
            "action_input": "Reached maximum reasoning steps",
            "thought": "Max steps reached"
        }
    
        return state
    
    # Build conversation context
    llm = get_llm("llama3", temperature=0.1)
    
    # Add system prompt if first step
    if step == 0:
        messages = [SystemMessage(content=react_system_prompt)] + messages
    else:
        # Re-emphasize format on each step
        format_reminder = """
CRITICAL REMINDER: You MUST use this exact format:

Thought: [your reasoning]
Action: [tool_name or Finish]
Action Input: [query or answer]

Do NOT write anything else. Follow the format strictly.
"""
        messages.append(SystemMessage(content=format_reminder))
    
    # ✅ INVOKE LLM 
    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    log("react", f"Step {step + 1} output:\n{response_text[:200]}...")
    
    # Parse ReAct format
    parsed = parse_react_output(response_text)
    
    # ✅ IMPROVED FALLBACK
    if not parsed["action"]:
        log("react", "⚠️ Failed to parse action, attempting fallback")
        
        response_lower = response_text.lower()
        
        # Check for web search indicators
        if "web" in response_lower or "latest" in response_lower or "current" in response_lower:
            parsed["action"] = "search_web_medical"
            # Extract query from question
            if not parsed["action_input"]:
                parsed["action_input"] = state["question"]
        
        # Check for database search indicators
        elif "database" in response_lower or "definition" in response_lower:
            parsed["action"] = "search_medical_database"
            if not parsed["action_input"]:
                parsed["action_input"] = state["question"]
        
        # Check for finish indicators
        elif "finish" in response_lower:
            parsed["action"] = "Finish"
            parsed["action_input"] = response_text
        
        # Default fallback
        else:
            log("react", "⚠️ No action detected, defaulting to database search")
            parsed["action"] = "search_medical_database"
            parsed["action_input"] = state["question"]
    
    # ✅ STORE PARSED ACTION IN STATE (for router to use)
    state["last_action"] = parsed
    
    # Add to reasoning trace
    trace_entry = f"""🧠 **Thought {step + 1}:** {parsed['thought']}
🎬 **Action:** {parsed['action']}
📝 **Input:** {parsed['action_input']}
"""
    
    if "reasoning_trace" not in state:
        state["reasoning_trace"] = []
    state["reasoning_trace"].append(trace_entry.strip())
    
    # Increment step
    state["step_count"] = step + 1
    
    # Add AI response to messages
    state["messages"].append(AIMessage(content=response_text))
    
    log("react", f"Stored action: {parsed['action']}, Input: {parsed['action_input'][:50]}...")
    
    return state


# ============================================================================
# NODE: Execute Tools
# ============================================================================

def execute_tools(state: State) -> State:
    """
    Execute tools with ONE-TIME forced web search.
    """
    messages = state["messages"]
    last_action = state.get("last_action", {})
    action = last_action.get("action", "").lower()
    action_input = last_action.get("action_input", "")
    
    # ✅ CRITICAL: Check if we already forced web search
    if "forced_web_search" not in state:
        state["forced_web_search"] = False
    
    # ✅ Only intercept Finish if we haven't forced yet
    if "finish" in action and not state["forced_web_search"]:
        question_lower = state["question"].lower()
        used_tools = state.get("used_tools", [])
        
        # 🚨 ‼️ CHECK: USE LLM/ WHY SEARCH QUERY HAS "2025 LATEST ?"
        requires_web = any([
            "this year" in question_lower,
            "new guidelines" in question_lower,
            "who release" in question_lower,
            "latest" in question_lower,
            "recent" in question_lower,
            "2025" in question_lower
        ])
        
        # If web needed and not done, FORCE ONCE
        if requires_web and "web" not in used_tools:
            log("tools", "🌐 FORCING web search (one-time only)")
            
            words = question_lower.replace("?", "").split()
            keywords = [w for w in words if len(w) > 3 and w not in ['what', 'should', 'person', 'with', 'this', 'year']]
            
            search_query = " ".join(keywords[:5]) + " 2025 latest"
            
            action = "search_web_medical"
            action_input = search_query
            
            # ✅ MARK THAT WE FORCED - don't force again!
            state["forced_web_search"] = True
            
            state["last_action"] = {
                "action": "search_web_medical",
                "action_input": search_query,
                "thought": "Forced web search"
            }
        else:
            # Normal finish - either no web needed or already forced
            state["draft"] = action_input if action_input and action_input != "None" else ""
            log("tools", "Finish accepted")
            return state
    elif "finish" in action:
        # We already forced, accept finish this time
        state["draft"] = action_input if action_input and action_input != "None" else ""
        log("tools", "Finish accepted (already forced)")
        return state
    
    log("tools", f"Executing: {action} with input: {action_input[:50]}...")
    
    # Initialize tracking
    if "used_tools" not in state:
        state["used_tools"] = []
    if "full_observations" not in state:
        state["full_observations"] = []
    if "tool_history" not in state:
        state["tool_history"] = []
    
    # Duplicate detection
    current_call = f"{action}|{action_input[:50]}"  # Use more chars for better matching

    if current_call in state.get("tool_history", []):
        log("tools", f"🚫 DUPLICATE DETECTED - breaking loop!")
        
        # If repeating database and web not used, FORCE web search
        if "database" in action and "web" not in state.get("used_tools", []):
            log("tools", "→ FORCING web search to break loop")
            
            # Extract temporal keywords from question
            question_lower = state["question"].lower()
            keywords = []
            for word in question_lower.split():
                if len(word) > 4 and word not in ['what', 'about', 'causes']:
                    keywords.append(word)
            
            action = "search_web_medical"
            action_input = " ".join(keywords[:5]) + " outbreak advisory 2025"
            
            # Update last_action
            state["last_action"] = {
                "action": "search_web_medical",
                "action_input": action_input,
                "thought": "Forced web search to break duplicate loop"
            }
            
            # Clear the duplicate from history so this new search can proceed
            # Don't add to history yet - will be added after successful execution
            
        else:
            # Already used both tools or repeating web search - just finish
            log("tools", "→ FORCING FINISH - duplicate detected")
            
            state["draft"] = "FORCED_FINISH_DUE_TO_DUPLICATE"
            state["last_action"] = {
                "action": "Finish",
                "action_input": "Forcing finish due to duplicate",
                "thought": "Breaking loop"
            }
            return state
    
    
    state["tool_history"].append(current_call)
    
    # Match tool
    matched_tool = None
    if "web" in action:
        matched_tool = "search_web_medical"
    elif "database" in action or ("medical" in action and "web" not in action):
        matched_tool = "search_medical_database"
    else:
        matched_tool = "search_medical_database"
    
    log("tools", f"✓ Matched: {matched_tool}")
    
    # Execute
    observation = ""
    
    if matched_tool == "search_medical_database":
        retriever = docsearch.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(action_input)
        state["contexts"] = docs
        observation = search_medical_database.invoke(action_input)
        
        if "database" not in state["used_tools"]:
            state["used_tools"].append("database")
    
    elif matched_tool == "search_web_medical":
        try:
            observation = search_web_medical.invoke(action_input)
            state["contexts"] = []
            
            if "web" not in state["used_tools"]:
                state["used_tools"].append("web")
        except Exception as e:
            log("tools", f"⚠️ Web error: {str(e)}")
            observation = "Web search failed."
    
    # Store
    state["full_observations"].append(observation)
    
    trace_entry = f"""👁️ **Observation:** {observation[:300]}...
{'-'*40}
"""
    state["reasoning_trace"].append(trace_entry.strip())
    
    tools_used = ", ".join(state["used_tools"])
    # Build observation message with clear next step instructions
    used_tools = state.get("used_tools", [])
    question_lower = state["question"].lower()

    # Check if question needs web search
    needs_web = any(word in question_lower for word in ["advisory", "outbreak", "this month", "released", "who", "latest"])

    if needs_web and "web" not in used_tools:
        next_instruction = """
    You found medical facts, but the question also asks about CURRENT information (advisories, outbreaks, releases).

    You MUST now search the web:
    Thought: I have medical facts but need current WHO advisory information
    Action: search_web_medical
    Action Input: WHO dengue advisory outbreak 2025
    """
    else:
        next_instruction = """
    You now have all the information needed.

    Provide your final answer:
    Thought: I have all the information to answer both parts of the question
    Action: Finish
    Action Input: [Write complete answer addressing causes AND current advisories]
    """

    obs_message = f"""Observation from {matched_tool}:

    {observation[:600]}...

    Tools already used: {", ".join(used_tools)}
    {next_instruction}
    """
    state["messages"].append(HumanMessage(content=obs_message))
    
    log("tools", f"Added {len(observation)} chars")
    
    return state

# ============================================================================
# NODE: Generate Final Response
# ============================================================================

def generate_response(state: State) -> State:
    """
    Generate final polished response with references.
    SIMPLIFIED for reliability.
    """
    log("generate", "=" * 50)
    log("generate", "STARTING GENERATION")
    log("generate", f"Draft exists: {bool(state.get('draft'))}")
    log("generate", f"Draft length: {len(state.get('draft', ''))}")
    log("generate", f"Full observations count: {len(state.get('full_observations', []))}")
    log("generate", "=" * 50)
    
    draft = state.get("draft", "")
    
    # If draft already exists from Finish action, use it
    if draft and len(draft) > 50:
        log("generate", f"Using draft from Finish action: {len(draft)} chars")
        final_answer = draft
    else:
        log("generate", "No draft, extracting from observations")
        
        # Get all observations from full_observations
        observations = state.get("full_observations", [])
        
        if not observations:
            log("generate", "⚠️ No observations found!")
            final_answer = "I couldn't gather enough information to answer your question. Please try rephrasing."
        else:
            # Combine all observations
            combined_obs = "\n\n---\n\n".join(observations[:3])  # Max 3
            
            log("generate", f"Synthesizing from {len(observations)} observations")
            
            # Generate answer using MedGemma
            llm = get_llm("alibayram/medgemma:4b", temperature=0.2)
            
            synthesis_prompt = f"""Answer this question based on the information below.

            Question: {state['question']}

            Information:
            {combined_obs[:1500]}

            Provide a clear 3-4 sentence answer:"""
            
            try:
                response = llm.invoke(synthesis_prompt)
                final_answer = response.content if hasattr(response, 'content') else str(response)
                log("generate", f"Generated answer: {len(final_answer)} chars")
            except Exception as e:
                log("generate", f"⚠️ Generation error: {str(e)}")
                final_answer = "I apologize, but I encountered an error generating the response. Please try again."
    
    # Extract references from full_observations
    references = []
    seen = set()
    
    for obs in state.get("full_observations", []):
        # Database sources (📖)
        if "📖" in obs:
            db_refs = re.findall(r'📖\s*([^\n]{20,100})', obs)
            for ref in db_refs[:2]:  # Max 2 per observation
                clean_ref = ref.strip()
                if clean_ref not in seen:
                    references.append(f"📖 {clean_ref}")
                    seen.add(clean_ref)
        
        # Web sources (🌐) - extract title and URL
        if "🌐" in obs:
            # Match pattern: 🌐 Title\nURL\n
            web_pattern = r'🌐\s*([^\n]+)\n(https?://[^\s\n]+)'
            web_matches = re.findall(web_pattern, obs)
            
            for title, url in web_matches[:2]:  # Max 2 per observation
                title = title.strip()[:80]  # Truncate long titles
                ref_key = f"{title}|{url}"
                if ref_key not in seen:
                    references.append(f"🌐 [{title}]({url})")
                    seen.add(ref_key)
        
        # Stop if we have enough
        if len(references) >= 6:
            break
    
    log("generate", f"Extracted {len(references)} references")
    
    # Build final formatted response
    final_text = ""

    # Add reasoning trace if enabled
    if state.get("show_reasoning", True):
        final_text += "## 🔍 Reasoning Process\n\n"

        # Clean up reasoning traces - remove extra newlines
        for trace in state.get("reasoning_trace", []):
            # Remove excessive newlines and format properly
            cleaned_trace = trace.strip()
            if cleaned_trace:
                final_text += cleaned_trace + "\n"

        final_text += "\n---\n\n## 📋 Final Answer\n\n"

    # Add the answer
    final_text += final_answer.strip()

    # Add references section with better formatting
    if references:
        final_text += "\n\n---\n\n## 📚 Sources\n\n"
        for ref in references:
            final_text += f"{ref}\n"  # Remove bullet point, just direct listing

    # Add disclaimer
    final_text += "\n\n---\n\n**⚠️ Disclaimer:** This information is for educational purposes only. Please consult a qualified healthcare professional for personal medical advice."
    
    # Store in state
    state["draft"] = final_text
    
    log("generate", f"✅ Complete answer ready: {len(final_text)} chars")
    log("generate", f"Answer preview: {final_text[:200]}...")
    
    return state


# ============================================================================
# NODE: Safety Check
# ============================================================================

def check_safety(state: State) -> State:
    """Check if response is medically safe."""
    
    draft = state.get("draft", "")
    question = state.get("question", "")
    
    safety_llm = get_llm("alibayram/medgemma:4b", temperature=0.0)
    
    prompt = safety_prompt.format(q=question, text=draft)
    response = safety_llm.invoke(prompt)
    raw = response.content.upper() if hasattr(response, 'content') else str(response).upper()
    
    log("safety", f"Result: {raw}")
    
    # Determine safety
    if "UNSAFE" in raw:
        state["safety_score"] = 0.3
        state["draft"] = (
            "⚠️ I cannot provide a safe medical answer. "
            "Please consult a qualified healthcare professional."
        )
        log("safety", "⚠️ UNSAFE response detected")
    else:
        state["safety_score"] = 0.9
        log("safety", "✅ SAFE")
    
    return state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue(state: State) -> Literal["tools", "generate", "end"]:
    """
    Smart routing with LLM-based temporal detection.
    """
    messages = state["messages"]
    if not messages:
        return "end"
    
    parsed = state.get("last_action", {})
    action_lower = parsed.get("action", "").lower()
    step_count = state.get("step_count", 0)
    
    if "used_tools" not in state:
        state["used_tools"] = []
    
    log("router", f"Action: '{parsed.get('action')}', step: {step_count}/{MAX_REASONING_STEPS}")
    
    # Check max steps FIRST
    if step_count >= MAX_REASONING_STEPS:
        log("router", f"Max steps reached → generate")
        return "generate"
    
    if state.get("draft"):
        log("router", "Draft exists → generate")
        return "generate"
    
    # If Finish, check if we should force web search
    if "finish" in action_lower:
        question_lower = state["question"].lower()
        used_tools = state.get("used_tools", [])
        
        # ✅ USE LLM TO DETECT TEMPORAL QUESTIONS
        requires_web = needs_current_info(state["question"])
        
        if requires_web and "web" not in used_tools:
            log("router", "Finish detected but web search needed → tools")
            return "tools"
        
        log("router", "Finish accepted → generate")
        return "generate"
    
    # Valid action → execute
    if parsed.get("action") and parsed.get("action_input"):
        log("router", "Valid action → tools")
        return "tools"
    
    log("router", "No action → generate")
    return "generate"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """Build ReAct graph."""
    
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("react", react_reasoning)
    graph.add_node("tools", execute_tools)
    graph.add_node("generate", generate_response)
    graph.add_node("safety", check_safety)
    
    # Set entry
    graph.set_entry_point("react")
    
    # Add edges
    graph.add_conditional_edges(
        "react",
        should_continue,
        {
            "tools": "tools",
            "generate": "generate",
            "end": END
        }
    )
    
    # After tools → back to react
    graph.add_edge("tools", "react")
    
    # After generate → safety
    graph.add_edge("generate", "safety")
    
    # After safety → END
    graph.add_edge("safety", END)
    
    return graph.compile()


# ============================================================================
# EXPORT
# ============================================================================

agentic_rag = build_graph()

print("=" * 70)
print("✅ True ReAct Agentic RAG with Visible Reasoning compiled!")
print("=" * 70)
print("\nFeatures:")
print("  ✓ Explicit Thought → Action → Observation traces")
print("  ✓ User can see reasoning process")
print("  ✓ Automatic tool execution")
print("  ✓ Two-stage response generation (MedGemma + LLaMA)")
print("  ✓ Multi-source synthesis with dedicated prompt")
print("  ✓ Safety checking")
print("  ✓ Max 5 reasoning steps")
print("=" * 70)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    state = {
        "messages": [HumanMessage(content="What is diabetes and what are the latest treatments in 2025?")],
        "question": "What is diabetes and what are the latest treatments in 2025?",
        "reasoning_trace": [],
        "step_count": 0,
        "contexts": [],
        "draft": "",
        "grounded_score": 0.0,
        "safety_score": 0.0,
        "show_reasoning": True  # Set False to hide reasoning from user
    }
    
    result = agentic_rag.invoke(state)
    
    print("\n" + "=" * 70)
    print("FINAL OUTPUT:")
    print("=" * 70)
    print(result["draft"])