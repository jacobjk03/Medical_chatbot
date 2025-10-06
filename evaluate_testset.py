from src.evaluation import evaluate_response
from src.graph_app import agentic_rag  # your compiled pipeline

# Small evaluation dataset
test_set = [
    {
        "question": "What are the symptoms of diabetes?",
        "reference": "Common symptoms of diabetes include frequent urination, increased thirst, fatigue, and blurred vision."
    },
    {
        "question": "What is hypertension?",
        "reference": "Hypertension, also known as high blood pressure, is a condition where the force of blood against the artery walls is consistently too high."
    },
    {
        "question": "How is asthma usually treated?",
        "reference": "Asthma is usually treated with inhalers that open the airways and reduce inflammation, such as bronchodilators and corticosteroids."
    },
    {
        "question": "What are the causes of anemia?",
        "reference": "Anemia is commonly caused by blood loss, iron deficiency, chronic diseases, or problems with red blood cell production."
    },
    {
        "question": "What is a migraine?",
        "reference": "A migraine is a neurological condition characterized by intense headaches, often accompanied by nausea, sensitivity to light, and vision changes."
    },
    {
        "question": "What are the symptoms of flu?",
        "reference": "Flu symptoms include fever, chills, cough, sore throat, runny or stuffy nose, body aches, and fatigue."
    },
    {
        "question": "How can dehydration be prevented?",
        "reference": "Dehydration can be prevented by drinking enough water, especially in hot weather or during exercise, and by avoiding excessive alcohol or caffeine."
    },
    {
        "question": "What is pneumonia?",
        "reference": "Pneumonia is an infection of the lungs that causes inflammation in the air sacs, which may fill with fluid or pus."
    },
    {
        "question": "What are common symptoms of COVID-19?",
        "reference": "Common COVID-19 symptoms include fever, cough, loss of taste or smell, fatigue, and shortness of breath."
    },
    {
        "question": "What is cholesterol?",
        "reference": "Cholesterol is a waxy substance found in your blood that is needed to build healthy cells, but high levels can increase the risk of heart disease."
    }
]


results = []
for sample in test_set:
    init_state = {
        "question": sample["question"],
        "history": []
    }
    final_state = agentic_rag.invoke(init_state)
    candidate = final_state["draft"]

    metrics = evaluate_response(sample["reference"], candidate)
    print(f"Q: {sample['question']}")
    print(f"Generated: {candidate[:200]}...")
    print("Metrics:", metrics)
    print("-"*60)

    results.append(metrics)

# Compute average scores
avg_metrics = {k: sum(m[k] for m in results)/len(results) for k in results[0]}
print("Average metrics:", avg_metrics)

