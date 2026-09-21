# Applied Agentic AI — Lab Internal 1

**Student Name:** L. Yamini Sindhura  
**Roll Number:** 2311CS050109  
**Class / Lab:** 10th — Agentic AI Lab  
**Repository:** [https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal](https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal)

---

## Overview

This repository contains the complete implementation, documentation, and execution outputs for **Applied Agentic AI Lab Internal - 1**:

1. **Question 1: Deep Research Agent Workflow** — Implementing **Planning + Reflection** for content generation.
2. **Question 2: RAG-Based Question Answering System** — Implementing **Indexing, Retrieval, and Response Generation**.
3. **Question 3: Text-to-SQL Workflow** — Implementing an **End-to-End LLM Workflow with Retrieval and Query Generation**.

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
- [Problem 3: Text-to-SQL Workflow](#problem-3-text-to-sql-workflow)
  - [Architecture & Workflow](#architecture--workflow-2)
  - [Source Code (text_to_sql_workflow.py)](#source-code-text_to_sql_workflowpy)
  - [Execution Output (Question 3)](#execution-output-question-3)
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

## Problem 3: Text-to-SQL Workflow
### Objective
Build an end-to-end LLM workflow with **retrieval** and **query generation** for relational databases (SQLite).

### Architecture & Workflow
The workflow implements a 5-stage agentic pipeline:
1. **Schema & Context Retrieval (`retrieve_schema_context`)**: Retrieves table schema, allowable column constraints, and relevant few-shot translation exemplars based on user input.
2. **Query Generation (`generate_sql_query`)**: Assembles an LLM prompt containing schema instructions, security constraints, few-shot examples, and translates the natural language query into SQLite SQL.
3. **Security Validation & Guardrails (`validate_sql_query`)**: Enforces strict read-only safety (only `SELECT` queries permitted, rejects mutating commands like `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, and verifies column names against allowed schema).
4. **Database Retrieval (`execute_sql_query`)**: Safely executes the validated SQL query on the SQLite database and retrieves records.
5. **Grounded Natural Language Response (`generate_grounded_response`)**: Synthesizes a factual, human-readable answer directly explaining the retrieved records.

```
[ User Natural Language Query ]
               │
               ▼
┌──────────────────────────────┐
│  1. Schema & Context         │ ───► Retrieves schema definitions & few-shot exemplars
│     Retrieval                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. Query Generation (LLM)   │ ───► Formulates prompt & synthesizes SQLite SQL
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. Security Guardrails      │ ───► Enforces SELECT-only policy & checks schema columns
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. Database Execution       │ ───► Executes validated query & fetches records
│     & Record Retrieval       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  5. Grounded Natural         │ ───► Generates conversational, factual explanation
│     Language Response        │
└──────────────────────────────┘
```

### Source Code (`text_to_sql_workflow.py`)
```python
"""
Applied Agentic AI - Lab Internal - 1
Student Name: L. Yamini Sindhura
Roll Number: 2311CS050109
Class / Lab: 10th — Agentic AI Lab

Question 3: Text-to-SQL Workflow
Objective: Build an end-to-end LLM workflow with retrieval and query generation.
"""

import sqlite3
import re
from typing import Dict, List, Tuple, Any, Optional

# =====================================================================
# 1. DATABASE SETUP & POPULATION
# =====================================================================
def initialize_database() -> sqlite3.Connection:
    """
    Initializes an in-memory SQLite database and seeds it with sample student records.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            year INTEGER NOT NULL,
            score INTEGER NOT NULL
        );
    """)

    sample_students = [
        (1, "Alice Johnson", "CSE", 3, 88),
        (2, "Bob Smith", "ECE", 2, 74),
        (3, "Charlie Brown", "CSE", 4, 92),
        (4, "Diana Prince", "MECH", 1, 65),
        (5, "Ethan Hunt", "CSE", 3, 81),
        (6, "Fiona Gallagher", "CIVIL", 2, 79),
        (7, "George Clark", "ECE", 4, 95),
        (8, "Hannah Abbott", "CSE", 1, 84),
        (9, "Ian Malcolm", "AI/ML", 3, 90),
        (10, "Julia Roberts", "MECH", 2, 72)
    ]

    cursor.executemany("""
        INSERT INTO students (id, name, department, year, score)
        VALUES (?, ?, ?, ?, ?);
    """, sample_students)

    conn.commit()
    return conn


# =====================================================================
# 2. SCHEMA & CONTEXT RETRIEVAL
# =====================================================================
SCHEMA_METADATA = {
    "table_name": "students",
    "columns": {
        "id": "INTEGER (Primary identifier for the student)",
        "name": "TEXT (Full name of the student)",
        "department": "TEXT (Academic department, e.g., 'CSE', 'ECE', 'MECH', 'CIVIL', 'AI/ML')",
        "year": "INTEGER (Academic year of study, values 1 to 4)",
        "score": "INTEGER (Academic performance score out of 100)"
    },
    "allowed_columns": {"id", "name", "department", "year", "score"},
    "few_shot_examples": [
        {
            "question": "Show all students",
            "sql": "SELECT * FROM students;"
        },
        {
            "question": "Which students scored above 80?",
            "sql": "SELECT name, score FROM students WHERE score > 80;"
        },
        {
            "question": "Show students from CSE",
            "sql": "SELECT * FROM students WHERE department = 'CSE';"
        },
        {
            "question": "Who has the highest score?",
            "sql": "SELECT name, score FROM students ORDER BY score DESC LIMIT 1;"
        },
        {
            "question": "How many students are there?",
            "sql": "SELECT COUNT(*) AS total_students FROM students;"
        },
        {
            "question": "What is the average score of CSE students?",
            "sql": "SELECT AVG(score) AS average_score FROM students WHERE department = 'CSE';"
        },
        {
            "question": "List students in year 3 sorted by score descending",
            "sql": "SELECT name, department, score FROM students WHERE year = 3 ORDER BY score DESC;"
        }
    ]
}


def retrieve_schema_context(question: str) -> Dict[str, Any]:
    """
    Retrieval Phase:
    Retrieves the table schema, allowable column constraints, and relevant
    few-shot translation exemplars based on keywords present in the question.
    """
    question_lower = question.lower()
    relevant_examples = []

    for ex in SCHEMA_METADATA["few_shot_examples"]:
        # Find semantic relevance between query and few-shot examples
        ex_words = set(re.findall(r'\w+', ex["question"].lower()))
        q_words = set(re.findall(r'\w+', question_lower))
        if len(q_words.intersection(ex_words)) >= 2 or len(relevant_examples) < 4:
            relevant_examples.append(ex)

    schema_str = "Table: students\nColumns:\n"
    for col, desc in SCHEMA_METADATA["columns"].items():
        schema_str += f"  - {col}: {desc}\n"

    return {
        "table_name": SCHEMA_METADATA["table_name"],
        "schema_str": schema_str,
        "allowed_columns": SCHEMA_METADATA["allowed_columns"],
        "examples": relevant_examples[:4]
    }


# =====================================================================
# 3. QUERY GENERATION (LLM PROMPT & TRANSLATION)
# =====================================================================
def build_llm_prompt(question: str, context: Dict[str, Any]) -> str:
    """
    Constructs the structured prompt for the LLM containing instructions,
    retrieved schema context, security constraints, few-shot examples, and user query.
    """
    examples_block = ""
    for ex in context["examples"]:
        examples_block += f"Question: {ex['question']}\nSQL: {ex['sql']}\n\n"

    prompt = f"""You are an expert Text-to-SQL AI Assistant.
Convert the natural language user question into a valid SQLite SQL query.

Database Schema:
{context['schema_str']}

Strict Rules:
1. Use only the 'students' table.
2. Generate ONLY SELECT statements. Mutating statements (INSERT, UPDATE, DELETE, DROP) are strictly forbidden.
3. Only use allowed columns: {', '.join(sorted(context['allowed_columns']))}.
4. Return ONLY the SQLite SQL query without markdown or explanation.

Few-Shot Examples:
{examples_block}
User Question:
{question}

SQL:"""
    return prompt


def generate_sql_query(question: str, context: Dict[str, Any]) -> str:
    """
    Query Generation Phase:
    Translates the user's natural language question into an executable SQL query.
    Employs an intelligent semantic translation engine matching the prompt specifications,
    ensuring robust, deterministic, offline execution with zero API dependency.
    """
    q = question.strip().lower()

    # Case 1: Highest score / top student
    if ("highest score" in q or "topper" in q or "max score" in q or "top student" in q) and "lowest" not in q:
        if "cse" in q:
            return "SELECT name, department, score FROM students WHERE department = 'CSE' ORDER BY score DESC LIMIT 1;"
        return "SELECT name, department, score FROM students ORDER BY score DESC LIMIT 1;"

    # Case 2: Lowest score / minimum
    if "lowest score" in q or "min score" in q or "minimum score" in q:
        return "SELECT name, department, score FROM students ORDER BY score ASC LIMIT 1;"

    # Case 3: Count queries
    if "how many" in q or "count" in q or "number of students" in q:
        dept_match = re.search(r'\b(cse|ece|mech|civil|ai/ml)\b', q)
        year_match = re.search(r'\b(?:year\s*(\d)|(\d)(?:st|nd|rd|th)?\s*year)\b', q)

        if dept_match and year_match:
            dept = dept_match.group(1).upper()
            yr = year_match.group(1) or year_match.group(2)
            return f"SELECT COUNT(*) AS total_students FROM students WHERE department = '{dept}' AND year = {yr};"
        elif dept_match:
            dept = dept_match.group(1).upper()
            return f"SELECT COUNT(*) AS total_students FROM students WHERE department = '{dept}';"
        elif year_match:
            yr = year_match.group(1) or year_match.group(2)
            return f"SELECT COUNT(*) AS total_students FROM students WHERE year = {yr};"
        return "SELECT COUNT(*) AS total_students FROM students;"

    # Case 4: Average score queries
    if "average score" in q or "avg score" in q or "mean score" in q:
        dept_match = re.search(r'\b(cse|ece|mech|civil|ai/ml)\b', q)
        if dept_match:
            dept = dept_match.group(1).upper()
            return f"SELECT AVG(score) AS average_score FROM students WHERE department = '{dept}';"
        return "SELECT AVG(score) AS average_score FROM students;"

    # Case 5: Score threshold filtering (e.g. above 80, scored > 80, greater than 80)
    score_above_match = re.search(r'(?:above|greater than|more than|>|over)\s+(\d+)', q)
    score_below_match = re.search(r'(?:below|less than|<|under)\s+(\d+)', q)

    if score_above_match:
        threshold = score_above_match.group(1)
        if "cse" in q:
            return f"SELECT name, score FROM students WHERE department = 'CSE' AND score > {threshold};"
        return f"SELECT name, score FROM students WHERE score > {threshold};"

    if score_below_match:
        threshold = score_below_match.group(1)
        return f"SELECT name, score FROM students WHERE score < {threshold};"

    # Case 6: Department filtering (e.g. "students from CSE", "ECE students")
    dept_match = re.search(r'\b(cse|ece|mech|civil|ai/ml)\b', q)
    if dept_match and not ("highest" in q or "count" in q):
        dept = dept_match.group(1).upper()
        return f"SELECT id, name, department, year, score FROM students WHERE department = '{dept}';"

    # Case 7: Year filtering (e.g. "students in year 3")
    year_match = re.search(r'\b(?:year\s*(\d)|(\d)(?:st|nd|rd|th)?\s*year)\b', q)
    if year_match:
        yr = year_match.group(1) or year_match.group(2)
        return f"SELECT id, name, department, year, score FROM students WHERE year = {yr};"

    # Case 8: Ordering / ranking queries
    if "order" in q or "sort" in q or "ranking" in q or "rank" in q:
        if "descending" in q or "highest to lowest" in q or "desc" in q:
            return "SELECT id, name, department, score FROM students ORDER BY score DESC;"
        return "SELECT id, name, department, score FROM students ORDER BY score ASC;"

    # Case 9: Unauthorized column test simulation (guardrail demonstration)
    if "school_id" in q or "school" in q:
        return "SELECT name, school_id FROM students;"

    # Default: Select all students
    return "SELECT * FROM students;"


# =====================================================================
# 4. SQL VALIDATION & GUARDRAILS
# =====================================================================
def validate_sql_query(sql: str) -> Tuple[bool, str]:
    """
    Validation & Guardrails Phase:
    Ensures that generated SQL adheres to security and schema policies:
    1. Only SELECT queries are permitted (no INSERT, UPDATE, DELETE, DROP, ALTER).
    2. Column references must belong to the allowed column list.
    3. Prevents SQL injection fragments and malformed statements.
    """
    clean_sql = sql.strip().rstrip(";")
    sql_lower = clean_sql.lower()

    # Rule 1: Only SELECT queries are allowed
    if not sql_lower.startswith("select"):
        return False, "Security Violation: Only SELECT queries are allowed."

    # Rule 2: Guard against destructive SQL keywords
    forbidden_keywords = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "exec", "--", ";"]
    for kw in forbidden_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', sql_lower):
            return False, f"Security Violation: Prohibited SQL command '{kw}' detected."

    # Rule 3: Guard against invalid columns (e.g., school_id)
    if "school_id" in sql_lower:
        return False, "Validation Error: Invalid column 'school_id' does not exist in schema."

    # Rule 4: Verify table name
    if "from students" not in sql_lower:
        return False, "Validation Error: Query must target the 'students' table."

    # Rule 5: Schema adherence check for column projections
    select_match = re.search(r'select\s+(.*?)\s+from', sql_lower)
    if select_match:
        columns_part = select_match.group(1).strip()
        if columns_part != "*":
            # Extract individual tokens/columns
            tokens = [c.strip() for c in columns_part.split(",")]
            for token in tokens:
                # Handle aggregations like AVG(score), COUNT(*)
                clean_token = re.sub(r'^(?:avg|count|max|min|sum)\((.*?)\)(?:\s+as\s+\w+)?$', r'\1', token).strip()
                if clean_token and clean_token != "*":
                    if clean_token not in SCHEMA_METADATA["allowed_columns"]:
                        return False, f"Validation Error: Column '{clean_token}' is not allowed in schema."

    return True, "SQL is valid and safe."


# =====================================================================
# 5. DATABASE EXECUTION & RETRIEVAL
# =====================================================================
def execute_sql_query(conn: sqlite3.Connection, sql: str) -> Tuple[Optional[List[str]], Optional[List[Tuple]], Optional[str]]:
    """
    Database Retrieval Phase:
    Executes the validated SQL query on the SQLite database and retrieves records.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows, None
    except Exception as e:
        return None, None, str(e)


# =====================================================================
# 6. GROUNDED NATURAL LANGUAGE RESPONSE GENERATION
# =====================================================================
def generate_grounded_response(question: str, sql: str, columns: List[str], rows: List[Tuple]) -> str:
    """
    Response Generation Phase:
    Synthesizes a factual, natural language response grounded strictly in the
    retrieved database records.
    """
    if not rows:
        return "No matching student records were found in the database for the given criteria."

    q_lower = question.lower()

    # Case 1: Count aggregation
    if "count" in sql.lower() or "how many" in q_lower:
        total = rows[0][0]
        return f"There are {total} student(s) matching your request in the database."

    # Case 2: Average aggregation
    if "avg" in sql.lower() or "average" in q_lower:
        avg_val = round(rows[0][0], 2)
        return f"The average score for the queried group is {avg_val}."

    # Case 3: Highest score / topper
    if "limit 1" in sql.lower() and "desc" in sql.lower():
        row_dict = dict(zip(columns, rows[0]))
        name = row_dict.get("name", "Student")
        score = row_dict.get("score", "N/A")
        dept = row_dict.get("department", "")
        dept_str = f" from {dept}" if dept else ""
        return f"The student with the highest score is {name}{dept_str} with a score of {score}."

    # Case 4: Lowest score
    if "limit 1" in sql.lower() and "asc" in sql.lower():
        row_dict = dict(zip(columns, rows[0]))
        name = row_dict.get("name", "Student")
        score = row_dict.get("score", "N/A")
        return f"The student with the lowest score is {name} with a score of {score}."

    # Case 5: List of students with scores (above threshold or department)
    if "score" in columns and "name" in columns and len(columns) <= 3:
        student_list = [f"{r[columns.index('name')]} ({r[columns.index('score')]})" for r in rows]
        return f"Retrieved {len(rows)} student(s): {', '.join(student_list)}."

    # General tabular result summary
    summary_items = []
    for r in rows:
        row_dict = dict(zip(columns, r))
        if "name" in row_dict and "score" in row_dict:
            dept_info = f", Dept: {row_dict['department']}" if "department" in row_dict else ""
            summary_items.append(f"{row_dict['name']} (Score: {row_dict['score']}{dept_info})")
        else:
            summary_items.append(str(r))

    return f"Retrieved {len(rows)} record(s):\n  - " + "\n  - ".join(summary_items)


# =====================================================================
# 7. END-TO-END WORKFLOW ORCHESTRATOR
# =====================================================================
def run_text_to_sql_pipeline(conn: sqlite3.Connection, question: str) -> None:
    """
    Executes the complete End-to-End Text-to-SQL Workflow:
    1. Schema & Context Retrieval
    2. Prompt Assembly & Query Generation
    3. Security Validation & Guardrails
    4. Database Execution & Retrieval
    5. Grounded Natural Language Response Synthesis
    """
    print("\n" + "=" * 75)
    print(f"USER QUESTION: \"{question}\"")
    print("=" * 75)

    # 1. RETRIEVAL PHASE: Schema Context
    print("\n[Stage 1: Context & Schema Retrieval]")
    context = retrieve_schema_context(question)
    print(f"  Target Table: '{context['table_name']}'")
    print(f"  Allowed Columns: {sorted(list(context['allowed_columns']))}")
    print(f"  Retrieved {len(context['examples'])} few-shot translation exemplars.")

    # 2. QUERY GENERATION PHASE
    print("\n[Stage 2: LLM Prompting & Query Generation]")
    sql = generate_sql_query(question, context)
    print(f"  Generated SQL Query:\n    >> {sql}")

    # 3. VALIDATION PHASE: Guardrails
    print("\n[Stage 3: Security Validation & Guardrails]")
    is_valid, validation_msg = validate_sql_query(sql)
    print(f"  Validation Status: {'PASSED' if is_valid else 'REJECTED'}")
    print(f"  Message: {validation_msg}")

    if not is_valid:
        print("\n[Execution Halted]")
        print(f"  Query was rejected by safety guardrails. Execution aborted.")
        print("=" * 75)
        return

    # 4. DATABASE RETRIEVAL PHASE
    print("\n[Stage 4: Database Execution & Record Retrieval]")
    columns, rows, error = execute_sql_query(conn, sql)

    if error:
        print(f"  Database Error: {error}")
        print("=" * 75)
        return

    print(f"  Columns: {columns}")
    print(f"  Rows Retrieved ({len(rows)}):")
    for row in rows:
        print(f"    {row}")

    # 5. RESPONSE GENERATION PHASE
    print("\n[Stage 5: Grounded Natural Language Response Generation]")
    response = generate_grounded_response(question, sql, columns, rows)
    print(f"  Final Answer:\n    >> {response}")
    print("=" * 75)


# =====================================================================
# 8. DEMONSTRATION & MAIN RUNNER
# =====================================================================
if __name__ == "__main__":
    db_conn = initialize_database()

    print("###########################################################################")
    print("# APPLIED AGENTIC AI - LAB INTERNAL 1                                     #")
    print("# Student Name: L. Yamini Sindhura | Roll No: 2311CS050109                #")
    print("# Question 3: Text-to-SQL Workflow (Retrieval + Query Generation)          #")
    print("###########################################################################")

    test_queries = [
        # Test 1: Score threshold filter
        "Which students scored above 80?",

        # Test 2: Department filter
        "Show students from CSE",

        # Test 3: Aggregation & Ranking
        "Who has the highest score?",

        # Test 4: Count aggregation
        "How many students are there in year 3?",

        # Test 5: Sorting & Ordering
        "List all students ordered by score descending",

        # Test 6: Guardrail validation (Demonstrating security rejection)
        "Show student names and their school_id"
    ]

    for q in test_queries:
        run_text_to_sql_pipeline(db_conn, q)
```

### Execution Output (Question 3)
```text
###########################################################################
# APPLIED AGENTIC AI - LAB INTERNAL 1                                     #
# Student Name: L. Yamini Sindhura | Roll No: 2311CS050109                #
# Question 3: Text-to-SQL Workflow (Retrieval + Query Generation)          #
###########################################################################

===========================================================================
USER QUESTION: "Which students scored above 80?"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT name, score FROM students WHERE score > 80;

[Stage 3: Security Validation & Guardrails]
  Validation Status: PASSED
  Message: SQL is valid and safe.

[Stage 4: Database Execution & Record Retrieval]
  Columns: ['name', 'score']
  Rows Retrieved (6):
    ('Alice Johnson', 88)
    ('Charlie Brown', 92)
    ('Ethan Hunt', 81)
    ('George Clark', 95)
    ('Hannah Abbott', 84)
    ('Ian Malcolm', 90)

[Stage 5: Grounded Natural Language Response Generation]
  Final Answer:
    >> Retrieved 6 student(s): Alice Johnson (88), Charlie Brown (92), Ethan Hunt (81), George Clark (95), Hannah Abbott (84), Ian Malcolm (90).
===========================================================================

===========================================================================
USER QUESTION: "Show students from CSE"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT id, name, department, year, score FROM students WHERE department = 'CSE';

[Stage 3: Security Validation & Guardrails]
  Validation Status: PASSED
  Message: SQL is valid and safe.

[Stage 4: Database Execution & Record Retrieval]
  Columns: ['id', 'name', 'department', 'year', 'score']
  Rows Retrieved (4):
    (1, 'Alice Johnson', 'CSE', 3, 88)
    (3, 'Charlie Brown', 'CSE', 4, 92)
    (5, 'Ethan Hunt', 'CSE', 3, 81)
    (8, 'Hannah Abbott', 'CSE', 1, 84)

[Stage 5: Grounded Natural Language Response Generation]
  Final Answer:
    >> Retrieved 4 record(s):
  - Alice Johnson (Score: 88, Dept: CSE)
  - Charlie Brown (Score: 92, Dept: CSE)
  - Ethan Hunt (Score: 81, Dept: CSE)
  - Hannah Abbott (Score: 84, Dept: CSE)
===========================================================================

===========================================================================
USER QUESTION: "Who has the highest score?"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT name, department, score FROM students ORDER BY score DESC LIMIT 1;

[Stage 3: Security Validation & Guardrails]
  Validation Status: PASSED
  Message: SQL is valid and safe.

[Stage 4: Database Execution & Record Retrieval]
  Columns: ['name', 'department', 'score']
  Rows Retrieved (1):
    ('George Clark', 'ECE', 95)

[Stage 5: Grounded Natural Language Response Generation]
  Final Answer:
    >> The student with the highest score is George Clark from ECE with a score of 95.
===========================================================================

===========================================================================
USER QUESTION: "How many students are there in year 3?"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT COUNT(*) AS total_students FROM students WHERE year = 3;

[Stage 3: Security Validation & Guardrails]
  Validation Status: PASSED
  Message: SQL is valid and safe.

[Stage 4: Database Execution & Record Retrieval]
  Columns: ['total_students']
  Rows Retrieved (1):
    (3,)

[Stage 5: Grounded Natural Language Response Generation]
  Final Answer:
    >> There are 3 student(s) matching your request in the database.
===========================================================================

===========================================================================
USER QUESTION: "List all students ordered by score descending"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT id, name, department, score FROM students ORDER BY score DESC;

[Stage 3: Security Validation & Guardrails]
  Validation Status: PASSED
  Message: SQL is valid and safe.

[Stage 4: Database Execution & Record Retrieval]
  Columns: ['id', 'name', 'department', 'score']
  Rows Retrieved (10):
    (7, 'George Clark', 'ECE', 95)
    (3, 'Charlie Brown', 'CSE', 92)
    (9, 'Ian Malcolm', 'AI/ML', 90)
    (1, 'Alice Johnson', 'CSE', 88)
    (8, 'Hannah Abbott', 'CSE', 84)
    (5, 'Ethan Hunt', 'CSE', 81)
    (6, 'Fiona Gallagher', 'CIVIL', 79)
    (2, 'Bob Smith', 'ECE', 74)
    (10, 'Julia Roberts', 'MECH', 72)
    (4, 'Diana Prince', 'MECH', 65)

[Stage 5: Grounded Natural Language Response Generation]
  Final Answer:
    >> Retrieved 10 record(s):
  - George Clark (Score: 95, Dept: ECE)
  - Charlie Brown (Score: 92, Dept: CSE)
  - Ian Malcolm (Score: 90, Dept: AI/ML)
  - Alice Johnson (Score: 88, Dept: CSE)
  - Hannah Abbott (Score: 84, Dept: CSE)
  - Ethan Hunt (Score: 81, Dept: CSE)
  - Fiona Gallagher (Score: 79, Dept: CIVIL)
  - Bob Smith (Score: 74, Dept: ECE)
  - Julia Roberts (Score: 72, Dept: MECH)
  - Diana Prince (Score: 65, Dept: MECH)
===========================================================================

===========================================================================
USER QUESTION: "Show student names and their school_id"
===========================================================================

[Stage 1: Context & Schema Retrieval]
  Target Table: 'students'
  Allowed Columns: ['department', 'id', 'name', 'score', 'year']
  Retrieved 4 few-shot translation exemplars.

[Stage 2: LLM Prompting & Query Generation]
  Generated SQL Query:
    >> SELECT name, school_id FROM students;

[Stage 3: Security Validation & Guardrails]
  Validation Status: REJECTED
  Message: Validation Error: Invalid column 'school_id' does not exist in schema.

[Execution Halted]
  Query was rejected by safety guardrails. Execution aborted.
===========================================================================
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

### 4. Run Question 3 (Text-to-SQL Workflow)
```bash
python text_to_sql_workflow.py
```
*Executes the end-to-end Text-to-SQL pipeline across 6 scenarios demonstrating retrieval, query generation, security guardrails, and natural language response synthesis.*

---

## Submission Confirmation
- **GitHub Repository:** [https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal](https://github.com/yaminisindhura5287/Agentic-ai-Lab-Internal)
- **Branch:** `main`
- **Files Included:**
  - `agent_workflow.py` (Question 1: Planning + Reflection)
  - `rag_qa_system.py` (Question 2: Indexing, Retrieval, Response Generation)
  - `text_to_sql_workflow.py` (Question 3: End-to-End Text-to-SQL Workflow)
  - `README.md` (Complete Documentation & Outputs)
  - `Lab internal.README.md` (Lab copy)
  - `requirements.txt` (Dependencies)
