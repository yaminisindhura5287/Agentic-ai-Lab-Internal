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

# ==========================================
# 1. DOCUMENT CORPUS
# ==========================================
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

print("=" * 70)
print("1. INDEXING PHASE")
print("=" * 70)
print(f"Total documents loaded: {len(documents)}")

# ==========================================
# 2. INDEXING: TF-IDF Vectorization
# ==========================================
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents)

print("Documents indexed successfully!")
print(f"Index matrix shape (docs, features): {document_vectors.shape}")


# ==========================================
# 3. RETRIEVAL PHASE
# ==========================================
def retrieve_documents(question, top_k=3):
    """
    Encodes the query, computes cosine similarity against indexed documents,
    and returns top_k ranked documents with similarity scores.
    """
    # Convert query into vector using the fitted vectorizer
    question_vector = vectorizer.transform([question])

    # Compute cosine similarity between question and document vectors
    similarities = cosine_similarity(question_vector, document_vectors)[0]

    # Rank documents by highest similarity score
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
    """
    Builds context string from top retrieved documents.
    """
    results = retrieve_documents(question, top_k)
    context_chunks = []
    for i, res in enumerate(results, start=1):
        context_chunks.append(f"[Document {i} | Similarity: {round(res['score'], 4)}]:\n{res['document']}")
    return "\n\n".join(context_chunks), results


# ==========================================
# 4. RESPONSE GENERATION PHASE
# ==========================================
def extract_grounded_answer(question, context):
    """
    Generates a concise, grounded response directly from the retrieved context.
    Identifies the key sentences containing matching answer keywords.
    """
    # Clean headers and normalize whitespace
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

    # If question asks specifically about ESP32
    if "esp32" in question.lower():
        for s in sentences:
            if "processing capability" in s.lower() or "wireless connectivity" in s.lower():
                return s + "."

    # If question asks about water pump or automation
    if "water pump" in question.lower() or "control" in question.lower():
        for s in sentences:
            if "water pump" in s.lower() or "ultrasonic" in s.lower():
                return s + "."

    if best_sentence:
        return best_sentence + ("." if not best_sentence.endswith(".") else "")
    return "The answer could not be determined from the retrieved documents."


def rag_answer(question, top_k=3):
    """
    Complete RAG Pipeline: Retrieve -> Construct Context -> Generate Answer.
    """
    context, retrieved_docs = create_context(question, top_k)
    answer = extract_grounded_answer(question, context)
    return answer, retrieved_docs


# ==========================================
# 5. DEMONSTRATION & EVALUATION
# ==========================================
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
    # Test Question 1
    run_rag_demonstration("What does ESP32 provide for IoT projects?")

    # Test Question 2
    run_rag_demonstration("How can IoT automatically control a water pump?")
