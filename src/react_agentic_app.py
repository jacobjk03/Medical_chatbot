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
    safety_prompt
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
    Determine if question requires current/recent information.
    Uses fast keyword check first, then LLM if needed.
    """
    # Cache to avoid repeated checks
    if not hasattr(needs_current_info, 'cache'):
        needs_current_info.cache = {}
    
    if question in needs_current_info.cache:
        return needs_current_info.cache[question]
    
    # ✅ CHECK KEYWORDS FIRST (fast and accurate for obvious cases)
    if keyword_temporal_check(question):
        needs_current_info.cache[question] = True
        log("router", f"Temporal check (keyword): {question[:50]}... → CURRENT")
        return True
    
    # ✅ If keywords didn't catch it, use LLM for ambiguous cases
    try:
        classifier_llm = get_llm("llama3.2:1b", temperature=0.0)
        
        prompt = f"""Does this question ask about NEW, RECENT, or CURRENT information (2024-2025) and FUTURE OR appears as if a follow up to previos question?

Question: {question}

Answer ONLY 'YES' or 'NO':
- YES: new guidelines, recent advisories, latest updates, current outbreaks, published this year
- NO: general medical facts, definitions, established symptoms, historical information

Answer:"""
        
        response = classifier_llm.invoke(prompt)
        answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        result = "YES" in answer
        needs_current_info.cache[question] = result
        
        log("router", f"Temporal check (LLM): {question[:50]}... → {'CURRENT' if result else 'GENERAL'}")
        return result
        
    except Exception as e:
        log("router", f"Temporal check LLM failed: {e}, defaulting to False")
        # Conservative default: assume not temporal if LLM fails
        needs_current_info.cache[question] = False
        return False


def keyword_temporal_check(question: str) -> bool:
    """
    Enhanced keyword-based temporal detection.
    Returns True if question needs current/recent information.
    """
    q_lower = question.lower()
    
    # ✅ CRITICAL TEMPORAL KEYWORDS - high confidence indicators
    critical_keywords = [
        # New/Recent indicators
        "new", "recent", "latest", "current", "updated", "nowadays", "these days",
        
        # Publishing/Release actions
        "publish", "published", "release", "released", "announce", "announced",
        "issue", "issued", "recommend", "recommended",
        
        # Time references
        "this year", "this month", "this week", "today", "yesterday", "last week",
        "right now", "currently", "presently",
        
        # Years
        "2024", "2025", "2026",
        
        # Events requiring current data
        "outbreak", "advisory", "warning", "alert", "epidemic", "pandemic",
        
        # Trends
        "spreading", "trend", "trending", "rising", "increasing", "surge"
    ]
    
    # ✅ STRONG PHRASES - combinations that always mean temporal
    strong_phrases = [
        "did who", "did cdc", "did aha", "did fda",  # "did X publish/release"
        "has who", "has cdc", "has aha", "has fda",
        "new guideline", "new prevention", "new recommendation", "new treatment",
        "published guideline", "latest guideline", "recent study",
        "this month", "this year"
    ]
    
    # Check critical keywords
    if any(keyword in q_lower for keyword in critical_keywords):
        return True
    
    # Check strong phrases
    if any(phrase in q_lower for phrase in strong_phrases):
        return True
    
    # Date patterns (e.g., "in 2025", "October 2024")
    import re
    if re.search(r'\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', q_lower):
        return True
    
    return False

def is_follow_up_question(current_question: str, conversation_history: List[AnyMessage]) -> bool:
    """
    Use fast LLM to determine if question is a follow-up.
    Much more flexible than hardcoded rules.
    """
    # Need at least 2 messages
    if len(conversation_history) < 2:
        return False
    
    # Extract last exchange (last 2-4 messages)
    recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history[-2:]
    
    # Build compact conversation summary
    context_parts = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            content = msg.content[:150] if hasattr(msg, 'content') else ""
            context_parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            # Extract just the answer part, skip reasoning
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, str):
                # Skip reasoning traces
                if "## 📋 Final Answer" in content:
                    content = content.split("## 📋 Final Answer")[1]
                elif "##" in content:
                    # Take last section
                    content = content.split("##")[-1]
                content = content[:200]  # First 200 chars of answer
            context_parts.append(f"Assistant: {content}")
    
    context_text = "\n".join(context_parts)
    
    # Use fast LLM with very specific prompt
    try:
        classifier = get_llm("llama3.2:1b", temperature=0.0)
        
        prompt = f"""Look at this conversation and decide if the current question is a FOLLOW-UP.

Recent conversation:
{context_text}

Current question: {current_question}

A follow-up question:
- Asks for MORE details about the previous answer
- Uses "what were", "which ones", "those", "these", "tell me more"
- Refers to something just mentioned WITHOUT repeating full context
- Examples: "what were the guidelines?", "which countries?", "tell me more"

NOT a follow-up:
- Completely new topic
- Has full context in the question itself
- Examples: "what are symptoms of diabetes?", "tell me about heart disease"

Is the current question a FOLLOW-UP?
Answer ONLY: YES or NO"""

        response = classifier.invoke(prompt)
        answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        is_followup = "YES" in answer
        
        log("router", f"Follow-up check (LLM): '{current_question[:40]}...' → {'FOLLOW-UP' if is_followup else 'NEW'}")
        return is_followup
        
    except Exception as e:
        log("router", f"⚠️ LLM follow-up check failed: {e}, using heuristic fallback")
        # Fallback to simple heuristics
        return simple_heuristic_followup(current_question, conversation_history)


def simple_heuristic_followup(question: str, history: List[AnyMessage]) -> bool:
    """
    Simple fallback heuristics if LLM fails.
    Only catches the most obvious cases.
    """
    q_lower = question.lower()
    q_len = len(question.split())
    
    # Very obvious follow-ups
    obvious_indicators = [
        q_lower.startswith("what were") and q_len < 8,
        q_lower.startswith("which") and q_len < 6,
        q_lower.startswith(("those", "these", "that", "them")),
        "tell me more" in q_lower,
        "elaborate" in q_lower,
        q_len < 4  # Very short questions
    ]
    
    result = any(obvious_indicators)
    if result:
        log("router", f"Follow-up check (heuristic): '{question[:40]}...' → FOLLOW-UP")
    
    return result


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
        question_lower = state["question"].lower()
        needs_web = needs_current_info(state["question"])
    
        is_follow_up = False
        if len(messages) > 2:
            is_follow_up = is_follow_up_question(state["question"], messages)
    
        # ✅ CRITICAL: Enforce strict format with examples
        enhanced_prompt = react_system_prompt + """

    ⚠️⚠️⚠️ CRITICAL FORMAT REQUIREMENT ⚠️⚠️⚠️

    You MUST respond in this EXACT format. NO EXCEPTIONS:

    Thought: [one sentence about what you need to do]
    Action: [MUST be one of: search_medical_database, search_web_medical, or Finish]
    Action Input: [your search query OR final answer]

    CORRECT EXAMPLES:

    Example 1:
    Thought: I need general medical facts about diabetes.
    Action: search_medical_database
    Action Input: diabetes symptoms causes treatment

    Example 2:
    Thought: I need current information about WHO guidelines.
    Action: search_web_medical
    Action Input: WHO 2025 guidelines dengue

    Example 3:
    Thought: I have enough information to answer the question.
    Action: Finish
    Action Input: Diabetes is a chronic condition that affects how your body processes blood sugar...

    WRONG - DO NOT DO THIS:
    ❌ "Based on the information provided, diabetes is..."
    ❌ "The answer is..."
    ❌ "According to the data..."

    You MUST use the Thought/Action/Action Input format. DO NOT answer directly.
    """
    
        # Add specific guidance based on question type
        if is_follow_up and needs_web:
            enhanced_prompt += """

    🔔 THIS IS A FOLLOW-UP QUESTION:
    The user wants MORE DETAILS about your previous answer.

    Your FIRST response MUST be:
    Thought: This is a follow-up asking for specific details from previous answer.
    Action: search_web_medical
    Action Input: [specific topic they're asking about]

    Do NOT search database again - you already have general facts.
    """
    
        elif needs_web:
            enhanced_prompt += """

    🔔 THIS NEEDS CURRENT INFORMATION:
    The question asks about recent/new/published information.

    Your strategy:
    1. First: Search database for general medical facts
    2. Then: Search web for current updates
    3. Finally: Use Action: Finish with complete answer
    """
    
        elif is_follow_up:
            enhanced_prompt += """

    🔔 THIS IS A FOLLOW-UP QUESTION:
    The user wants MORE DETAILS about your previous answer.

    Search the database for more specific information about what they're asking.
    """
    
        messages = [SystemMessage(content=enhanced_prompt)] + messages
        log("react", f"Enhanced prompt: web={needs_web}, follow_up={is_follow_up}")

    else:
        # ✅ RE-EMPHASIZE FORMAT ON EVERY SUBSEQUENT STEP
        format_reminder = SystemMessage(content="""
    REMINDER - Use this EXACT format:

    Thought: [your reasoning]
    Action: [search_medical_database, search_web_medical, or Finish]
    Action Input: [query or answer]

    DO NOT write anything else. FOLLOW THE FORMAT EXACTLY.
    """)
        messages.append(format_reminder)
    
    # ✅ INVOKE LLM 
    response = llm.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    log("react", f"Step {step + 1} output:\n{response_text[:200]}...")
    
    # Parse ReAct format
    parsed = parse_react_output(response_text)
    
    # ✅ IMPROVED FALLBACK
    if not parsed["action"]:
        log("react", "⚠️ Failed to parse action - LLM did not follow format!")
    
        response_lower = response_text.lower()
        step = state.get("step_count", 0)
    
        # ✅ Check if LLM is answering directly (WRONG!)
        direct_answer_indicators = [
            response_text.startswith("Based on"),
            response_text.startswith("The answer"),
            response_text.startswith("According to"),
            "the information provided" in response_text[:150],
            len(response_text) > 200 and "Thought:" not in response_text
        ]
    
        if any(direct_answer_indicators):
            log("react", "❌ LLM answered directly instead of using ReAct format - forcing Finish")
        
            # If it answered directly, treat it as the final answer
            parsed["action"] = "Finish"
            parsed["action_input"] = response_text[:800]  # Use direct answer
            parsed["thought"] = "LLM provided direct answer"
    
        # If we've failed to parse multiple times, force finish
        elif step >= 2:
            log("react", "⚠️ Multiple parse failures - forcing Finish to avoid loop")
            parsed["action"] = "Finish"
            parsed["action_input"] = "I encountered formatting issues. Please rephrase your question."
            parsed["thought"] = "Format failures"
    
        # Otherwise try to infer action from content
        elif "web" in response_lower or "search" in response_lower[:100]:
            log("react", "Inferring: search_web_medical")
            parsed["action"] = "search_web_medical"
            parsed["action_input"] = state["question"]
    
        elif "database" in response_lower:
            log("react", "Inferring: search_medical_database")
            parsed["action"] = "search_medical_database"
            parsed["action_input"] = state["question"]
    
        else:
            # Last resort - search database
            log("react", "⚠️ No clear action detected - defaulting to database")
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
    Execute tools with duplicate prevention and clear guidance.
    """
    messages = state["messages"]
    last_action = state.get("last_action", {})
    action = last_action.get("action", "").lower()
    action_input = last_action.get("action_input", "")
    
    # Initialize tracking
    if "used_tools" not in state:
        state["used_tools"] = []
    if "full_observations" not in state:
        state["full_observations"] = []
    if "tool_history" not in state:
        state["tool_history"] = []
    if "forced_web_search" not in state:
        state["forced_web_search"] = False
    
    # ✅ INTERCEPT FINISH to force web search if needed
    if "finish" in action:
        if not state["forced_web_search"]:
            requires_web = needs_current_info(state["question"])
            
            # If web needed and not done yet, FORCE ONCE
            if requires_web and "web" not in state["used_tools"]:
                log("tools", "🌐 INTERCEPTING Finish - forcing web search")
                
                action = "search_web_medical"
                action_input = state["question"]
                state["forced_web_search"] = True
                
                state["last_action"] = {
                    "action": "search_web_medical",
                    "action_input": action_input,
                    "thought": "Forced web search for current information"
                }
                # Continue to execute web search below
            else:
                # Can finish now
                state["draft"] = action_input if action_input and action_input != "None" else ""
                log("tools", "Finish accepted")
                return state
        else:
            # Already forced once, allow finish
            state["draft"] = action_input if action_input and action_input != "None" else ""
            log("tools", "Finish accepted (already forced)")
            return state
    
    log("tools", f"Executing: {action} with input: {action_input[:50]}...")
    
    # ✅ DUPLICATE DETECTION - prevent infinite loops
    current_call = f"{action}|{action_input[:50]}"
    
    if current_call in state["tool_history"]:
        log("tools", f"🚫 DUPLICATE DETECTED: {action}")
        
        # If repeating database and haven't used web, switch to web
        if "database" in action and "web" not in state["used_tools"]:
            log("tools", "→ Switching to web search to break loop")
            
            action = "search_web_medical"
            action_input = state["question"]  # Use full question
            
            state["last_action"] = {
                "action": "search_web_medical",
                "action_input": action_input,
                "thought": "Switched to web due to duplicate database call"
            }
            # Don't add to history yet - will be added below after new action
            
        elif "web" in action:
            # Duplicate web search - force finish
            log("tools", "→ FORCING FINISH (duplicate web search)")
            
            observation = "Already searched web multiple times. Proceeding to final answer."
            state["full_observations"].append(observation)
            
            # Add directive message to force finish
            state["messages"].append(HumanMessage(content=f"""Observation: {observation}

🛑 You have searched both database and web. No more searches needed.

You MUST finish NOW:
Thought: I have complete information from all sources
Action: Finish
Action Input: [Write comprehensive answer using all gathered information]"""))
            
            return state
        else:
            # Other duplicate - force finish
            log("tools", "→ FORCING FINISH (unknown duplicate)")
            state["draft"] = "FORCED_FINISH_DUPLICATE"
            return state
    
    # Add current call to history
    state["tool_history"].append(current_call)
    
    # ✅ MATCH TOOL
    matched_tool = None
    if "web" in action or "internet" in action:
        matched_tool = "search_web_medical"
    elif "database" in action or ("medical" in action and "web" not in action):
        matched_tool = "search_medical_database"
    else:
        # Default to database
        matched_tool = "search_medical_database"
    
    log("tools", f"✓ Matched: {matched_tool}")
    
    # ✅ EXECUTE TOOL
    observation = ""
    
    if matched_tool == "search_medical_database":
        try:
            retriever = docsearch.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(action_input)
            state["contexts"] = docs
            observation = search_medical_database.invoke(action_input)
            
            if "database" not in state["used_tools"]:
                state["used_tools"].append("database")
        except Exception as e:
            log("tools", f"⚠️ Database error: {str(e)}")
            observation = "Database search failed."
    
    elif matched_tool == "search_web_medical":
        try:
            observation = search_web_medical.invoke(action_input)
            state["contexts"] = []
            
            if "web" not in state["used_tools"]:
                state["used_tools"].append("web")
        except Exception as e:
            log("tools", f"⚠️ Web error: {str(e)}")
            observation = "Web search failed."
    
    # ✅ STORE OBSERVATION
    state["full_observations"].append(observation)
    
    trace_entry = f"""👁️ **Observation:** {observation[:300]}...
{'-'*40}
"""
    state["reasoning_trace"].append(trace_entry.strip())
    
    # ✅ BUILD DIRECTIVE OBSERVATION MESSAGE
    used_tools = state["used_tools"]
    has_database = "database" in used_tools
    has_web = "web" in used_tools
    
    # Determine what to do next
    if has_database and has_web:
        # Has BOTH sources - MUST finish
        next_instruction = """

✅ **YOU HAVE BOTH SOURCES NOW** (medical database + web search)

You MUST finish immediately:

Thought: I have complete information from both medical database and current web sources
Action: Finish
Action Input: [Write comprehensive answer addressing all parts of the question]

DO NOT search again. FINISH NOW."""

    elif has_database and not has_web:
        # Has database, check if needs web
        needs_web = needs_current_info(state["question"])
        
        if needs_web:
            next_instruction = f"""

⚠️ You have general medical facts, but this question asks about CURRENT/RECENT information.

Your NEXT action MUST be:

Thought: I have background facts but need current updates from the web
Action: search_web_medical
Action Input: {state["question"]}

Focus on the temporal aspect: new guidelines, recent updates, current advisories."""

        else:
            next_instruction = """

✅ You have sufficient information from the medical database.

You MUST finish now:

Thought: I have all the medical information needed to answer the question
Action: Finish
Action Input: [Write clear answer based on database information]

DO NOT search again. FINISH NOW."""

    elif has_web and not has_database:
        # Has web only (rare case - usually follow-ups)
        next_instruction = """

✅ You have current information from web search.

You MUST finish now:

Thought: I have current web information to answer the question
Action: Finish
Action Input: [Write answer using the web search results]

DO NOT search again. FINISH NOW."""

    else:
        # No tools used yet (first search result)
        next_instruction = """

Continue with your search strategy based on the question type."""

    # Build final observation message
    obs_message = f"""Observation from {matched_tool}:

{observation[:600]}

Tools used so far: {", ".join(used_tools)}
{next_instruction}"""

    state["messages"].append(HumanMessage(content=obs_message))
    
    log("tools", f"Added observation ({len(observation)} chars)")
    
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
            combined_obs = "\n\n---\n\n".join(observations)  # Max 3
            
            log("generate", f"Synthesizing from {len(observations)} observations")
            
            # Generate answer using MedGemma
            llm = get_llm("alibayram/medgemma:4b", temperature=0.2)
            
            synthesis_prompt = f"""Answer this question based on the information below.

            Question: {state['question']}

            Information:
            {combined_obs}

            Provide a clear and brief answer:"""
            
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
    Smart routing with LLM-based follow-up detection.
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
        current_question = state["question"]
        question_lower = current_question.lower()
        used_tools = state.get("used_tools", [])
        
        # ✅ PRIMARY CHECK: Is this question temporal?
        requires_web = needs_current_info(current_question)
        
        # ✅ SECONDARY CHECK: Is this a follow-up to a temporal question?
        if not requires_web and len(messages) > 2:
            # Check if this is a follow-up using LLM
            is_follow_up = is_follow_up_question(current_question, messages)
            
            if is_follow_up:
                # Check if previous question used web search
                previous_used_web = False
                
                for msg in messages[-6:]:
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        msg_content = msg.content.lower()
                        if any(indicator in msg_content for indicator in [
                            "search_web_medical", "🌐", "advisory", "outbreak", 
                            "guideline", "latest", "published"
                        ]):
                            previous_used_web = True
                            break
                
                if previous_used_web:
                    log("router", "Follow-up to temporal question → forcing web")
                    requires_web = True
        
        # ✅ FORCE WEB SEARCH if needed
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
        "safety_score": 0.0,
        "show_reasoning": True  # Set False to hide reasoning from user
    }
    
    result = agentic_rag.invoke(state)
    
    print("\n" + "=" * 70)
    print("FINAL OUTPUT:")
    print("=" * 70)
    print(result["draft"])