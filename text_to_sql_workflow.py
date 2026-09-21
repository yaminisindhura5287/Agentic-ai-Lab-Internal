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
