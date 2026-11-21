"""Reproduce the baseline and current evidence gate without a model or Milvus."""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag_knowledge_base.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class FixtureDocument:
    id: str
    page_content: str


def character_features(text):
    words = "".join(character for character in text.casefold() if character.isalnum())
    return Counter(words) + Counter(words[i:i + 2] for i in range(len(words) - 1))


def cosine(left, right):
    denominator = math.sqrt(sum(value ** 2 for value in left.values()))
    denominator *= math.sqrt(sum(value ** 2 for value in right.values()))
    if not denominator:
        return 0.0
    return sum(value * right[key] for key, value in left.items()) / denominator


class FixtureSearch:
    def __init__(self, records):
        self.index = [
            (FixtureDocument(row["id"], row["text"]), character_features(row["text"]))
            for row in records
        ]

    def search(self, query, k, collection_name):
        features = character_features(query)
        scores = [(document, cosine(features, vector)) for document, vector in self.index]
        return sorted(scores, key=lambda row: (-row[1], row[0].id))[:k]


class RecordingRetriever(VectorRetriever):
    def _generate_answer_with_llm(self, question, context):
        return "Generation invoked"


def evaluate(fixture):
    backend = FixtureSearch(fixture["documents"])
    candidate = RecordingRetriever(backend)
    cases = []
    for row in fixture["questions"]:
        question = row["question"]
        # The source template kept scores >= 0.5 and called the LLM even with no matches.
        previous = [
            document.id for document, score in backend.search(question, 5, "fixture")
            if score >= 0.5
        ]
        result = candidate.answer_question(question, "fixture", k=5)
        cases.append({
            **row,
            "baseline_references": previous,
            "current_references": [document.id for document in result.source_documents],
            "current_status": result.status,
        })

    answerable = [case for case in cases if case["expected"]]
    unsupported = [case for case in cases if not case["expected"]]

    def recall(reference_field):
        return sum(
            len(set(case["expected"]) & set(case[reference_field])) / len(case["expected"])
            for case in answerable
        ) / len(answerable)

    return {
        "evaluation": "Offline character n-gram retrieval and simulated generation",
        "answerable_questions": len(answerable),
        "unsupported_questions": len(unsupported),
        "baseline": {
            "evidence_recall_at_5": recall("baseline_references"),
            "unsupported_questions_sent_to_generation": len(unsupported),
            "answerable_questions_sent_to_generation": len(answerable),
        },
        "current": {
            "evidence_recall_at_5": recall("current_references"),
            "unsupported_questions_sent_to_generation": sum(
                case["current_status"] == "answered" for case in unsupported
            ),
            "answerable_questions_sent_to_generation": sum(
                case["current_status"] == "answered" for case in answerable
            ),
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the full JSON report to this file")
    args = parser.parse_args()
    fixture = json.loads(Path(__file__).with_name("after_sales_fixture.json").read_text(encoding="utf-8"))
    report = evaluate(fixture)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
