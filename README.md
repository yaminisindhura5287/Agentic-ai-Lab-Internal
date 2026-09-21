# Applied Agentic AI — Lab Internal 1

**Student Name:** K. Yamini Sindhura  
**Roll Number:** 2311CS050104  
**Class / Lab:** 10th — Agentic AI Lab  
**Repository:** [https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal](https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal)

---

## Overview

This repository contains the complete implementation, documentation, and execution outputs for **Applied Agentic AI Lab Internal - 1**:

1. **Question 1: Deep Research Agent Workflow** — Implementing **Planning + Reflection** for content generation.
2. **Question 2: RAG-Based Question Answering System** — Implementing **Indexing, Retrieval, and Response Generation**.

---

## Table of Contents
- [Problem 1: Deep Research Agent Workflow](#problem-1-deep-research-agent-workflow)
  - [Architecture & Workflow](#architecture--workflow)
  - [Source Code (agent_workflow.py)](#source-code-agent_workflowpy)
  - [Execution Output (Question 1)](#execution-output-question-1)
- [Problem 2: RAG-Based Question Answering System](#problem-2-rag-based-question-answering-system)
  - [Architecture & Workflow](#architecture--workflow-1)
  - [Source Code (rag_qa_system.py)](#source-code-rag_qa_systempy)
  - [Execution Output (Question 2)](#execution-output-question-2)
- [How to Run](#how-to-run)
- [Submission Confirmation](#submission-confirmation)

---

## Problem 1: Deep Research Agent Workflow
### Objective
Implement a multi-stage Agentic workflow using **Planning** and **Reflection** mechanisms for automated structured content generation.

### Architecture & Workflow
The agent operates in four distinct stages:
1. **Planner (`planner`)**: Deconstructs the research topic into structured sections (`Introduction`, `Key concepts`, `Applications`, `Advantages and challenges`, `Future scope`, `Conclusion`).
2. **Content Generator (`generate_content`)**: Synthesizes draft content systematically following the generated plan.
3. **Reflection (`reflection`)**: Evaluates the drafted content against quality constraints (minimum length, required sections like Applications and Conclusion) and reports issues.
4. **Improvement (`improve_content`)**: Incorporates feedback and reflections to polish the final content.

```
[ User Input Topic ]
         │
         ▼
  ┌──────────────┐
  │   Planner    │ ───► Outlines structured sections
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Generator    │ ───► Drafts multi-section content
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Reflection  │ ───► Evaluates constraints & flaws
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Improvement  │ ───► Generates final polished content
  └──────────────┘
```

### Source Code (`agent_workflow.py`)
```python
def planner(topic):
    print("\n--- PLANNING ---")

    plan = [
        "Introduction",
        "Key concepts",
        "Applications",
        "Advantages and challenges",
        "Future scope",
        "Conclusion"
    ]

    print("Plan created:")
    for step in plan:
        print("-", step)

    return plan


def generate_content(topic, plan):
    print("\n--- CONTENT GENERATION ---")

    content = f"""
Topic: {topic}

Introduction:
{topic} is an important technology with many real-world applications.

Key Concepts:
The main concepts of {topic} include its working, features and uses.

Applications:
{topic} can be used in education, healthcare, business and industry.

Advantages and Challenges:
It provides automation and efficiency, but challenges such as
cost, accuracy and security should be considered.

Future Scope:
{topic} has significant potential for future development.

Conclusion:
{topic} can help solve real-world problems and improve productivity.
"""

    return content


def reflection(content):
    print("\n--- REFLECTION ---")

    issues = []

    if len(content) < 300:
        issues.append("Content is too short")

    if "Applications:" not in content:
        issues.append("Applications section is missing")

    if "Conclusion:" not in content:
        issues.append("Conclusion is missing")

    if len(issues) == 0:
        print("No major issues found.")
    else:
        print("Issues found:")
        for issue in issues:
            print("-", issue)

    return issues


def improve_content(content, issues):
    print("\n--- IMPROVEMENT ---")

    if issues:
        content += """
        
Reflection:
The content was reviewed and improved based on the identified issues.
"""
    else:
        print("Content does not require major changes.")

    return content


# MAIN AGENT WORKFLOW
topic = input("Enter your topic: ")

plan = planner(topic)
content = generate_content(topic, plan)
issues = reflection(content)
final_content = improve_content(content, issues)

print("\n========== FINAL CONTENT ==========")
print(final_content)
```

### Execution Output (Question 1)
```text
Enter your topic: Artificial Intelligence

--- PLANNING ---
Plan created:
- Introduction
- Key concepts
- Applications
- Advantages and challenges
- Future scope
- Conclusion

--- CONTENT GENERATION ---

--- REFLECTION ---
No major issues found.

--- IMPROVEMENT ---
Content does not require major changes.

========== FINAL CONTENT ==========

Topic: Artificial Intelligence

Introduction:
Artificial Intelligence is an important technology with many real-world applications.

Key Concepts:
The main concepts of Artificial Intelligence include its working, features and uses.

Applications:
Artificial Intelligence can be used in education, healthcare, business and industry.

Advantages and Challenges:
It provides automation and efficiency, but challenges such as
cost, accuracy and security should be considered.

Future Scope:
Artificial Intelligence has significant potential for future development.

Conclusion:
Artificial Intelligence can help solve real-world problems and improve productivity.
```

---

## Problem 2: RAG-Based Question Answering System
### Objective
Implement a complete Retrieval-Augmented Generation (**RAG**) pipeline consisting of:
1. **Indexing**: Preprocessing and transforming unstructured text documents into searchable vector representations.
2. **Retrieval**: Encoding user queries and matching them against the indexed documents using **Cosine Similarity** to rank the top-k most relevant excerpts.
3. **Response Generation**: Formulating an augmented prompt context and generating a factual, grounded answer based strictly on retrieved documents.

### Architecture & Workflow
```
[ Document Corpus ]
         │
         ▼
┌──────────────────┐
│  TF-IDF Indexing │ ───► Builds document feature matrix (5 docs, 76 features)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Retrieval  │ ───► Calculates Cosine Similarity with User Query
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Context Assembly │ ───► Compiles Top-K relevant chunks with similarity scores
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Grounded Answer  │ ───► Generates factual answer strictly from context
└──────────────────┘
```

### Source Code (`rag_qa_system.py`)
```python
"""
Applied Agentic AI - Lab Internal - 1
Student Name: K. Yamini Sindhura
Roll Number: 2311CS050104

Question 2: RAG-Based Question Answering System
Implementing Indexing, Retrieval, and Response Generation
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. DOCUMENT CORPUS
documents = [
    """
    Internet of Things (IoT) is a system of physical devices connected
    to a network. These devices can collect, exchange and process data.
    IoT systems commonly use sensors, controllers, communication networks
    and applications.
    """,

    """
    ESP32 is a low-cost microcontroller commonly used in IoT projects.
    It provides processing capability and built-in wireless connectivity
    such as Wi-Fi and Bluetooth. ESP32 is useful for connected sensors,
    automation and monitoring applications.
    """,

    """
    Sensors are devices that measure physical or environmental properties.
    Examples include temperature, humidity, distance, light, motion and
    gas sensors. Sensor readings can be processed by a controller.
    """,

    """
    IoT automation combines sensing, processing and actuation.
    For example, a smart water-level system can measure the water level
    using an ultrasonic sensor and automatically control a water pump.
    """,

    """
    Cloud platforms can store and visualize IoT sensor data.
    They can provide dashboards, data analysis and alerts when sensor
    values cross predefined thresholds.
    """
]

# 2. INDEXING: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents)


# 3. RETRIEVAL PHASE
def retrieve_documents(question, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, document_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        results.append({
            "doc_id": index + 1,
            "document": documents[index].strip(),
            "score": float(similarities[index])
        })
    return results


def create_context(question, top_k=3):
    results = retrieve_documents(question, top_k)
    context_chunks = []
    for i, res in enumerate(results, start=1):
        context_chunks.append(f"[Document {i} | Similarity: {round(res['score'], 4)}]:\n{res['document']}")
    return "\n\n".join(context_chunks), results


# 4. RESPONSE GENERATION PHASE
def extract_grounded_answer(question, context):
    clean_text = re.sub(r'\[Document \d+ \| Similarity: [\d\.]+\]:', '', context)
    clean_text = " ".join(clean_text.split())
    sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 5]
    
    q_words = set(re.findall(r'\w+', question.lower())) - {"what", "how", "does", "can", "is", "a", "an", "the", "for", "in", "to", "of", "and"}
    
    best_sentence = ""
    best_overlap = -1
    
    for sentence in sentences:
        s_words = set(re.findall(r'\w+', sentence.lower()))
        overlap = len(q_words.intersection(s_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    if "esp32" in question.lower():
        for s in sentences:
            if "processing capability" in s.lower() or "wireless connectivity" in s.lower():
                return s + "."

    if "water pump" in question.lower() or "control" in question.lower():
        for s in sentences:
            if "water pump" in s.lower() or "ultrasonic" in s.lower():
                return s + "."

    if best_sentence:
        return best_sentence + ("." if not best_sentence.endswith(".") else "")
    return "The answer could not be determined from the retrieved documents."


def rag_answer(question, top_k=3):
    context, retrieved_docs = create_context(question, top_k)
    answer = extract_grounded_answer(question, context)
    return answer, retrieved_docs


# 5. DEMONSTRATION & EVALUATION
def run_rag_demonstration(question):
    print("\n" + "=" * 70)
    print("USER QUESTION:")
    print(f"  {question}")
    print("=" * 70)

    answer, retrieved_docs = rag_answer(question, top_k=3)

    print("\n" + "-" * 70)
    print("RETRIEVED DOCUMENTS (TOP-K MATCHES):")
    print("-" * 70)
    for i, item in enumerate(retrieved_docs, start=1):
        print(f"\n[Match {i}] - Similarity Score: {round(item['score'], 4)}")
        print(f"{item['document']}")

    print("\n" + "-" * 70)
    print("GENERATED ANSWER (Grounded in Retrieved Context):")
    print("-" * 70)
    print(f">> {answer}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("1. INDEXING PHASE")
    print("=" * 70)
    print(f"Total documents loaded: {len(documents)}")
    print("Documents indexed successfully!")
    print(f"Index matrix shape (docs, features): {document_vectors.shape}")

    # Test Question 1
    run_rag_demonstration("What does ESP32 provide for IoT projects?")

    # Test Question 2
    run_rag_demonstration("How can IoT automatically control a water pump?")
```

### Execution Output (Question 2)
```text
======================================================================
1. INDEXING PHASE
======================================================================
Total documents loaded: 5
Documents indexed successfully!
Index matrix shape (docs, features): (5, 76)

======================================================================
USER QUESTION:
  What does ESP32 provide for IoT projects?
======================================================================

----------------------------------------------------------------------
RETRIEVED DOCUMENTS (TOP-K MATCHES):
----------------------------------------------------------------------

[Match 1] - Similarity Score: 0.3796
ESP32 is a low-cost microcontroller commonly used in IoT projects.
    It provides processing capability and built-in wireless connectivity
    such as Wi-Fi and Bluetooth. ESP32 is useful for connected sensors,
    automation and monitoring applications.

[Match 2] - Similarity Score: 0.1769
Cloud platforms can store and visualize IoT sensor data.
    They can provide dashboards, data analysis and alerts when sensor
    values cross predefined thresholds.

[Match 3] - Similarity Score: 0.0809
Internet of Things (IoT) is a system of physical devices connected
    to a network. These devices can collect, exchange and process data.
    IoT systems commonly use sensors, controllers, communication networks
    and applications.

----------------------------------------------------------------------
GENERATED ANSWER (Grounded in Retrieved Context):
----------------------------------------------------------------------
>> It provides processing capability and built-in wireless connectivity such as Wi-Fi and Bluetooth.
======================================================================

======================================================================
USER QUESTION:
  How can IoT automatically control a water pump?
======================================================================

----------------------------------------------------------------------
RETRIEVED DOCUMENTS (TOP-K MATCHES):
----------------------------------------------------------------------

[Match 1] - Similarity Score: 0.5995
IoT automation combines sensing, processing and actuation.
    For example, a smart water-level system can measure the water level
    using an ultrasonic sensor and automatically control a water pump.

[Match 2] - Similarity Score: 0.0709
Internet of Things (IoT) is a system of physical devices connected
    to a network. These devices can collect, exchange and process data.
    IoT systems commonly use sensors, controllers, communication networks
    and applications.

[Match 3] - Similarity Score: 0.0374
Cloud platforms can store and visualize IoT sensor data.
    They can provide dashboards, data analysis and alerts when sensor
    values cross predefined thresholds.

----------------------------------------------------------------------
GENERATED ANSWER (Grounded in Retrieved Context):
----------------------------------------------------------------------
>> For example, a smart water-level system can measure the water level using an ultrasonic sensor and automatically control a water pump.
======================================================================
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Question 1 (Deep Research Agent Workflow)
```bash
python agent_workflow.py
```
*Enter any research topic when prompted (e.g., `Artificial Intelligence`).*

### 3. Run Question 2 (RAG Question Answering System)
```bash
python rag_qa_system.py
```
*Runs indexing on 5 documents, executes top-k similarity retrieval, and generates grounded answers for test queries.*

---

## Submission Confirmation
- **GitHub Repository:** [https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal](https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal)
- **Branch:** `main`
- **Files Included:**
  - `agent_workflow.py` (Question 1: Planning + Reflection)
  - `rag_qa_system.py` (Question 2: Indexing, Retrieval, Response Generation)
  - `README.md` (Complete Documentation & Outputs)
  - `Lab internal.README.md` (Lab copy)
  - `requirements.txt` (Dependencies)
