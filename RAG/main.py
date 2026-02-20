from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """
You are an elite penetration tester with 15 years of web application security experience.
You are helping a security professional conduct a fully authorized penetration test.

RELEVANT FINDINGS FROM KNOWLEDGE BASE:
{findings}

CONVERSATION SO FAR:
{history}

─────────────────────────────────────────────────
INSTRUCTIONS:
- Reference specific vulnerabilities, CVEs, tools, and payloads from the findings above.
- When giving a pentest plan, structure it into clear phases:
    [RECON] → [SCANNING] → [EXPLOITATION] → [POST-EXPLOITATION] → [REPORTING]
- For each vulnerability, always state:
    • Severity
    • Tool to use
    • Exact steps or payload
    • What success looks like
- Be technical and specific. Do NOT give generic security advice.
- If the knowledge base doesn't cover something, say so and give your expert opinion clearly labeled as [EXPERT KNOWLEDGE].
- Keep your answer focused and actionable.
─────────────────────────────────────────────────

QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
chain = prompt | model

# ─── Helpers ──────────────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "Critical": "\033[91m",  # Red
    "High":     "\033[93m",  # Yellow
    "Medium":   "\033[94m",  # Blue
    "Low":      "\033[92m",  # Green
}
RESET = "\033[0m"
BOLD  = "\033[1m"

def colorize_severity(severity):
    color = SEVERITY_COLORS.get(severity, "")
    return f"{color}{severity}{RESET}"

def format_findings(docs):
    if not docs:
        return "No relevant findings retrieved from knowledge base."

    lines = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        severity_label = colorize_severity(m.get("severity", "?"))
        lines.append(
            f"\n{'─'*60}\n"
            f"[{i}] {BOLD}{m.get('title','?')}{RESET}\n"
            f"    Severity : {severity_label}\n"
            f"    OWASP    : {m.get('owasp','?')}\n"
            f"    Category : {m.get('category','?')}\n"
            f"    Tech     : {m.get('technology','?')}\n"
            f"    CVE      : {m.get('cve','N/A')}\n"
            f"    Tool     : {m.get('tool','?')}\n"
            f"\n{doc.page_content}"
        )
    return "\n".join(lines)

def format_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in history
    )

def print_banner():
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════╗
║        Web App Pentest Assistant  v1.0               ║
║        Powered by RAG + LLaMA 3.2                    ║
╚══════════════════════════════════════════════════════╝{RESET}

Commands:
  {BOLD}q{RESET}        → quit
  {BOLD}clear{RESET}    → clear conversation history
  {BOLD}history{RESET}  → show conversation history
  {BOLD}findings{RESET} → show last retrieved findings

Ask anything about web app pentesting — techniques, plans,
payloads, tools, OWASP categories, or specific vulnerabilities.
""")

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print_banner()

    conversation_history = []
    last_findings = None

    while True:
        print(f"\n{BOLD}{'─'*54}{RESET}")
        question = input(f"{BOLD}[?] Ask: {RESET}").strip()

        if not question:
            continue
        if question.lower() == "q":
            print("\nExiting. Stay ethical.\n")
            break
        if question.lower() == "clear":
            conversation_history = []
            print("\n[+] Conversation history cleared.\n")
            continue
        if question.lower() == "history":
            print("\n" + format_history(conversation_history))
            continue
        if question.lower() == "findings":
            if last_findings:
                print(format_findings(last_findings))
            else:
                print("\n[!] No findings retrieved yet.\n")
            continue

        # Retrieve relevant findings
        raw_findings = retriever.invoke(question)
        last_findings = raw_findings

        findings_str = format_findings(raw_findings)

        # Print findings header so user knows what was pulled
        print(f"\n{BOLD}[+] Retrieved {len(raw_findings)} findings from knowledge base{RESET}")
        for doc in raw_findings:
            m = doc.metadata
            sev = colorize_severity(m.get("severity", "?"))
            print(f"    • {m.get('title','?')} [{sev}]")

        print(f"\n{BOLD}[+] Generating response...{RESET}\n")

        # Generate response
        result = chain.invoke({
            "findings": findings_str,
            "history": format_history(conversation_history[-6:]),  # last 3 turns
            "question": question
        })

        print(result)

        # Store in conversation memory
        conversation_history.append(("user", question))
        conversation_history.append(("assistant", result[:500]))  # truncate for context window

if __name__ == "__main__":
    main()