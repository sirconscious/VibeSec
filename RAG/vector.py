from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("pentest_findings.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_pentest_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Embed ALL fields so queries on any term (CVE, tool, OWASP, payload) hit correctly
        page_content = f"""
Title: {row['Title']}
Severity: {row['Severity']}
Category: {row['Category']}
OWASP: {row['OWASP']}
Technology: {row['Technology']}
CVE: {row.get('CVE', 'N/A')}
Tool: {row['Tool']}
Description: {row['Description']}
Payload Example: {row['Payload_Example']}
Steps: {row['Steps']}
Success Indicator: {row['Success_Indicator']}
Remediation: {row['Remediation']}
""".strip()

        document = Document(
            page_content=page_content,
            metadata={
                "title": row["Title"],
                "severity": row["Severity"],
                "category": row["Category"],
                "owasp": row["OWASP"],
                "technology": row["Technology"],
                "cve": str(row.get("CVE", "")),
                "tool": row["Tool"],
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="pentest_findings",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 6,
        "score_threshold": 0.15
    }
)