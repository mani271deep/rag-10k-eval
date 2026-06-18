import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from generator import answer

load_dotenv()

JUDGE_MODEL = "gpt-4o"
K = 5
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

_judge = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are evaluating a RAG system's answer about SEC 10-K filings.

Question: {question}

Reference answer (ground truth): {reference}

Retrieved context the system was given:
{context}

System's answer: {system_answer}

Score two dimensions on a 1-5 scale and return ONLY a JSON object:
- "faithfulness": Is the answer grounded in the retrieved context, with no \
fabricated facts? 5 = fully grounded, 1 = hallucinated. If the system \
correctly abstained ("not enough information") AND the context genuinely \
lacks the answer, score 5.
- "correctness": Does the answer match the reference answer's key facts? \
5 = fully correct, 1 = wrong. For unanswerable questions, abstaining = 5, \
inventing an answer = 1.

Return exactly: {{"faithfulness": N, "correctness": N, "reason": "brief"}}"""


def judge(question, reference, context, system_answer):
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference,
        context=context[:6000], system_answer=system_answer)
    resp = _judge.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def retrieval_metrics(chunks, target_companies, target_sections):
    """Hit@k and reciprocal rank: a chunk is correct if its company is in
    target_companies AND its section is in target_sections."""
    rr = 0.0
    hit = 0
    for rank, c in enumerate(chunks, 1):
        if c["company"] in target_companies and c["section"] in target_sections:
            hit = 1
            rr = 1.0 / rank
            break
    return hit, rr


def main():
    gt = json.load(open(GT_PATH))
    results = []
    print(f"Running eval on {len(gt)} questions (k={K}, judge={JUDGE_MODEL})\n")

    for q in gt:
        out = answer(q["question"], k=K)
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answerable": q["answerable"],
            "system_answer": out["answer"],
            "retrieved": [f"{c['company']}/{c['section']}" for c in out["chunks"]],
        }
        if q["answerable"]:
            hit, rr = retrieval_metrics(
                out["chunks"], q["target_companies"], q["target_sections"])
            rec["hit@k"] = hit
            rec["reciprocal_rank"] = round(rr, 3)
        else:
            rec["hit@k"] = None
            rec["reciprocal_rank"] = None

        scores = judge(q["question"], q["reference_answer"],
                       out["context"], out["answer"])
        rec["faithfulness"] = scores["faithfulness"]
        rec["correctness"] = scores["correctness"]
        rec["judge_reason"] = scores.get("reason", "")
        results.append(rec)
        print(f"  {q['id']} ({q['type']}): "
              f"hit@k={rec['hit@k']} rr={rec['reciprocal_rank']} "
              f"faith={rec['faithfulness']} correct={rec['correctness']}")
        time.sleep(0.3)

    json.dump(results, open(RESULTS_PATH, "w"), ensure_ascii=False, indent=2)

    # --- aggregate ---
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    hit_rate = sum(r["hit@k"] for r in answerable) / len(answerable)
    mrr = sum(r["reciprocal_rank"] for r in answerable) / len(answerable)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_correct = sum(r["correctness"] for r in results) / len(results)
    abstain_correct = sum(1 for r in unanswerable if r["correctness"] >= 4)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Retrieval (answerable, n={len(answerable)}):")
    print(f"  Hit@{K}: {hit_rate:.1%}")
    print(f"  MRR:    {mrr:.3f}")
    print(f"Answer quality (all, n={len(results)}):")
    print(f"  Faithfulness (avg/5): {avg_faith:.2f}")
    print(f"  Correctness  (avg/5): {avg_correct:.2f}")
    print(f"Abstention (unanswerable, n={len(unanswerable)}):")
    print(f"  Correct abstentions: {abstain_correct}/{len(unanswerable)}")
    print(f"\nFull results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
