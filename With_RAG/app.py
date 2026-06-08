import pandas as pd
import gradio as gr
import re

# =========================
# LOAD CSV FILES
# =========================
model_df = pd.read_csv("6_model_responses.csv")
risk_df = pd.read_csv("5_risk_labels.csv")
eval_df = pd.read_csv("7_human_evaluation.csv")
benchmark_df = pd.read_csv("3_benchmark_questions.csv")
ideal_df = pd.read_csv("4_ideal_answers.csv")

# =========================
# CLEAN FUNCTION
# =========================
def clean(df):
    df.columns = df.columns.str.strip()
    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].astype(str).str.upper().str.strip()
    return df

model_df = clean(model_df)
risk_df = clean(risk_df)
eval_df = clean(eval_df)
benchmark_df = clean(benchmark_df)
ideal_df = clean(ideal_df)

benchmark_df["user_question"] = benchmark_df["user_question"].astype(str).str.lower()

# =========================
# FIND QUESTION (ID + TEXT)
# =========================
def find_question(user_input):

    user_input = user_input.strip().lower()

    # CASE 1: ID input (Q001 etc)
    if re.match(r"^q\d+", user_input, re.IGNORECASE):
        row = benchmark_df[benchmark_df["question_id"].str.lower() == user_input]
        if not row.empty:
            return row.iloc[0]

    # CASE 2: TEXT MATCH
    match = benchmark_df[
        benchmark_df["user_question"].str.contains(user_input, na=False)
    ]

    if not match.empty:
        return match.iloc[0]

    # CASE 3: FALLBACK LOOP
    for _, row in benchmark_df.iterrows():
        if user_input in str(row["user_question"]).lower():
            return row

    return None


# =========================
# RAG SYSTEM
# =========================
def rag_system(user_input):

    q_row = find_question(user_input)

    if q_row is None:
        return "❌ No matching question found in dataset"

    qid = str(q_row["question_id"]).upper()
    question_text = q_row["user_question"]

    # =========================
    # MODEL ANSWER
    # =========================
    model_row = model_df[model_df["question_id"] == qid]
    model_answer = model_row["response"].values[0] if not model_row.empty else "N/A"

    # =========================
    # SOURCE / IDEAL ANSWER
    # =========================
    ideal_row = ideal_df[ideal_df["question_id"] == qid]
    source_answer = ideal_row["ideal_answer"].values[0] if not ideal_row.empty else "N/A"

    # =========================
    # RISK LABEL
    # =========================
    risk_row = risk_df[risk_df["question_id"] == qid]
    risk = risk_row["risk_label"].values[0] if not risk_row.empty else "N/A"

    # =========================
    # HUMAN EVALUATION
    # =========================
    eval_row = eval_df[eval_df["question_id"] == qid]

    if not eval_row.empty:
        best_system = eval_row["system_type"].values[0]
        comment = eval_row["comments"].values[0]
    else:
        best_system = "N/A"
        comment = "N/A"

    # =========================
    # WINNER LOGIC
    # =========================
    if "chatgpt" in str(best_system).lower():
        winner = "🤖 ChatGPT is better"
    elif best_system != "N/A":
        winner = f"📚 {best_system} is better"
    else:
        winner = "No evaluation available"

    # =========================
    # FINAL OUTPUT
    # =========================
    return f"""
=========================
🧠 QUESTION
=========================
{question_text}

=========================
📚 SOURCE ANSWER (IDEAL CSV)
=========================
{source_answer}

=========================
🤖 MODEL ANSWER (CSV)
=========================
{model_answer}

=========================
⚠ RISK LABEL
=========================
{risk}

=========================
🏆 HUMAN EVALUATION
=========================
Best System: {best_system}

💬 Comment:
{comment}

📊 Result:
{winner}
"""


# =========================
# GRADIO UI
# =========================
demo = gr.Interface(
    fn=rag_system,
    inputs=gr.Textbox(label="Enter Question ID or Question Text"),
    outputs=gr.Textbox(label="RAG OUTPUT"),
    title="Career Anxiety FULL RAG + Evaluation System"
)

demo.launch()