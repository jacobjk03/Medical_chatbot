"""
Comprehensive Evaluation Testing for ReAct Medical Chatbot
Tests both answer quality and reasoning quality
"""

from src.evaluation import (
    evaluate_react_response,
    print_metric_explanations,
    METRIC_DESCRIPTIONS
)
from src.react_agentic_app import agentic_rag
from langchain_core.messages import HumanMessage
import json
from datetime import datetime


# ============================================================================
# ENHANCED TEST DATASET
# ============================================================================

test_set = [
    # Simple definition questions (should use database only)
    {
        "question": "What is diabetes?",
        "reference": "Diabetes is a chronic condition where blood sugar levels remain too high because the body cannot produce enough insulin or use it effectively.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    },
    {
        "question": "What is hypertension?",
        "reference": "Hypertension, also known as high blood pressure, is a condition where the force of blood against the artery walls is consistently too high.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    },
    {
        "question": "What is a migraine?",
        "reference": "A migraine is a neurological condition characterized by intense headaches, often accompanied by nausea, sensitivity to light, and vision changes.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    },
    
    # Questions about symptoms (database)
    {
        "question": "What are the symptoms of flu?",
        "reference": "Flu symptoms include fever, chills, cough, sore throat, runny or stuffy nose, body aches, and fatigue.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    },
    {
        "question": "What are the symptoms of diabetes?",
        "reference": "Common symptoms of diabetes include frequent urination, increased thirst, fatigue, and blurred vision.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    },
    
    # Latest/recent questions (should use web)
    {
        "question": "What are the latest treatments for diabetes in 2025?",
        "reference": "The latest diabetes treatments in 2025 include advanced GLP-1 receptor agonists, continuous glucose monitoring systems, and automated insulin delivery devices.",
        "expected_tools": ["search_web_medical"],
        "complexity": "moderate"
    },
    {
        "question": "What are recent advances in cancer treatment?",
        "reference": "Recent cancer treatment advances include immunotherapy, targeted therapy, CAR T-cell therapy, and precision medicine approaches.",
        "expected_tools": ["search_web_medical"],
        "complexity": "moderate"
    },
    
    # Compound questions (should use both tools)
    {
        "question": "What is asthma and what are the latest treatments?",
        "reference": "Asthma is a respiratory condition causing airway inflammation. Latest treatments include smart inhalers, biologics for severe asthma, and personalized treatment plans.",
        "expected_tools": ["search_medical_database", "search_web_medical"],
        "complexity": "complex"
    },
    {
        "question": "What is hypertension and how is it treated currently?",
        "reference": "Hypertension is high blood pressure. Current treatments include lifestyle modifications, ACE inhibitors, ARBs, calcium channel blockers, and combination therapies.",
        "expected_tools": ["search_medical_database", "search_web_medical"],
        "complexity": "complex"
    },
    
    # Treatment questions
    {
        "question": "How is asthma usually treated?",
        "reference": "Asthma is usually treated with inhalers that open the airways and reduce inflammation, such as bronchodilators and corticosteroids.",
        "expected_tools": ["search_medical_database"],
        "complexity": "simple"
    }
]


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on test set."""
    
    print("\n" + "=" * 80)
    print("REACT MEDICAL CHATBOT - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print(f"Test Set Size: {len(test_set)} questions")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    results = []
    detailed_results = []
    
    for idx, sample in enumerate(test_set, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {idx}/{len(test_set)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question']}")
        print(f"Complexity: {sample['complexity']}")
        print(f"Expected Tools: {', '.join(sample['expected_tools'])}")
        print(f"{'-'*80}")
        
        # Build initial state
        init_state = {
            "messages": [HumanMessage(content=sample["question"])],
            "question": sample["question"],
            "reasoning_trace": [],
            "step_count": 0,
            "contexts": [],
            "draft": "",
            "grounded_score": 0.0,
            "safety_score": 0.0,
            "show_reasoning": True
        }
        
        try:
            # Invoke ReAct system
            final_state = agentic_rag.invoke(init_state)
            candidate = final_state["draft"]
            reasoning_trace = final_state.get("reasoning_trace", [])
            
            # Separate reasoning from answer
            if "## 📋 Final Answer" in candidate:
                parts = candidate.split("## 📋 Final Answer")
                final_answer = parts[1].strip() if len(parts) > 1 else candidate
            else:
                final_answer = candidate
            
            # Show generated answer
            print(f"\nGenerated Answer (first 300 chars):")
            print(f"{final_answer[:300]}...")
            print(f"\nReasoning Steps: {final_state.get('step_count', 0)}")
            
            # Compute metrics
            metrics = evaluate_react_response(
                question=sample["question"],
                reference=sample["reference"],
                candidate=final_answer,
                reasoning_trace=reasoning_trace,
                final_state=final_state
            )
            
            # Print key metrics
            print(f"\n{'Key Metrics':^80}")
            print(f"{'-'*80}")
            print(f"{'Answer Quality':<40} {'Reasoning Quality':<40}")
            print(f"{'-'*80}")
            print(f"ROUGE-1: {metrics['rouge1_f1']:.3f}                          "
                  f"Thought Clarity: {metrics['thought_clarity']:.3f}")
            print(f"BERT F1: {metrics['bert_f1']:.3f}                          "
                  f"Efficiency: {metrics['reasoning_efficiency']:.3f}")
            print(f"Completeness: {metrics['answer_completeness']:.3f}                      "
                  f"Tool Accuracy: {metrics['tool_selection_accuracy']:.3f}")
            print(f"{'-'*80}")
            print(f"Overall Answer Quality: {metrics['overall_answer_quality']:.3f}")
            print(f"Overall Reasoning Quality: {metrics['overall_reasoning_quality']:.3f}")
            print(f"Overall System Quality: {metrics['overall_system_quality']:.3f}")
            print(f"Safety Score: {metrics['safety_score']:.3f}")
            
            # Store results
            results.append(metrics)
            detailed_results.append({
                "question": sample["question"],
                "complexity": sample["complexity"],
                "expected_tools": sample["expected_tools"],
                "generated_answer": final_answer[:200],
                "metrics": metrics,
                "step_count": final_state.get('step_count', 0)
            })
            
        except Exception as e:
            print(f"\n⚠️ ERROR: {str(e)}")
            print("Skipping this test case...")
            continue
        
        print(f"{'='*80}\n")
    
    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80 + "\n")
    
    if not results:
        print("No successful test cases.")
        return
    
    # Compute averages
    avg_metrics = {}
    for key in results[0].keys():
        values = [r[key] for r in results if key in r]
        avg_metrics[key] = sum(values) / len(values)
    
    # Print category-wise results
    print(f"{'ANSWER QUALITY METRICS':^80}")
    print(f"{'-'*80}")
    print(f"ROUGE-1 F1:           {avg_metrics['rouge1_f1']:.3f}")
    print(f"ROUGE-2 F1:           {avg_metrics['rouge2_f1']:.3f}")
    print(f"ROUGE-L F1:           {avg_metrics['rougeL_f1']:.3f}")
    print(f"BERT Precision:       {avg_metrics['bert_precision']:.3f}")
    print(f"BERT Recall:          {avg_metrics['bert_recall']:.3f}")
    print(f"BERT F1:              {avg_metrics['bert_f1']:.3f}")
    print(f"Answer Completeness:  {avg_metrics['answer_completeness']:.3f}")
    print(f"{'-'*80}")
    print(f"Overall Answer Quality: {avg_metrics['overall_answer_quality']:.3f}")
    
    print(f"\n{'REASONING QUALITY METRICS':^80}")
    print(f"{'-'*80}")
    print(f"Thought Clarity:      {avg_metrics['thought_clarity']:.3f}")
    print(f"Thought-Action Align: {avg_metrics['thought_action_alignment']:.3f}")
    print(f"Reasoning Efficiency: {avg_metrics['reasoning_efficiency']:.3f}")
    print(f"Avg Step Count:       {avg_metrics['step_count']:.1f}")
    print(f"{'-'*80}")
    print(f"Overall Reasoning Quality: {avg_metrics['overall_reasoning_quality']:.3f}")
    
    print(f"\n{'TOOL USAGE METRICS':^80}")
    print(f"{'-'*80}")
    print(f"Tool Selection Accuracy: {avg_metrics['tool_selection_accuracy']:.3f}")
    print(f"Tool Diversity:          {avg_metrics['tool_diversity']:.3f}")
    
    print(f"\n{'SAFETY & GROUNDING METRICS':^80}")
    print(f"{'-'*80}")
    print(f"Safety Score:         {avg_metrics['safety_score']:.3f}")
    print(f"Source Grounding:     {avg_metrics['source_grounding']:.3f}")
    print(f"Hallucination Risk:   {avg_metrics['hallucination_risk']:.3f} (lower better)")
    
    print(f"\n{'OVERALL SYSTEM PERFORMANCE':^80}")
    print(f"{'-'*80}")
    print(f"Overall System Quality: {avg_metrics['overall_system_quality']:.3f}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # COMPLEXITY-BASED BREAKDOWN
    # ========================================================================
    
    print(f"\n{'PERFORMANCE BY QUESTION COMPLEXITY':^80}")
    print(f"{'='*80}")
    
    complexity_groups = {'simple': [], 'moderate': [], 'complex': []}
    for result in detailed_results:
        complexity = result['complexity']
        if complexity in complexity_groups:
            complexity_groups[complexity].append(result['metrics'])
    
    for complexity, metrics_list in complexity_groups.items():
        if not metrics_list:
            continue
        
        avg_quality = sum(m['overall_system_quality'] for m in metrics_list) / len(metrics_list)
        avg_steps = sum(m['step_count'] for m in metrics_list) / len(metrics_list)
        
        print(f"\n{complexity.upper()} Questions (n={len(metrics_list)})")
        print(f"{'-'*80}")
        print(f"Avg System Quality: {avg_quality:.3f}")
        print(f"Avg Steps Taken:    {avg_steps:.1f}")
    
    print(f"\n{'='*80}\n")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"evaluation_results_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "test_set_size": len(test_set),
        "aggregate_metrics": avg_metrics,
        "detailed_results": detailed_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    return avg_metrics, detailed_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Show metric explanations first
    print_metric_explanations()
    
    # Run evaluation
    avg_metrics, detailed_results = run_comprehensive_evaluation()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)