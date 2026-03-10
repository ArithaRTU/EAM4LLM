import os
import json
import re
import asyncio
from typing import Any, List, Dict, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from ragas.metrics.collections import (
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
)

try:
    from ragas.metrics.collections import ResponseRelevancy
except ImportError:
    from ragas.metrics.collections import AnswerRelevancy as ResponseRelevancy


INPUT_FILE = ""
OUTPUT_FILE = ""

LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-large"

TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 10000

# Basic loading / utility
def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if not numeric.dropna().empty else float("nan")


def extract_metric_value(result: Any) -> Optional[float]:
    if result is None:
        return None

    if isinstance(result, (int, float)):
        return float(result)

    value = getattr(result, "value", None)
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(result, dict):
        dict_value = result.get("value")
        if isinstance(dict_value, (int, float)):
            return float(dict_value)

    try:
        return float(result)
    except Exception:
        return None


async def safe_score_collections(metric, **kwargs) -> Optional[float]:
    try:
        raw_result = await metric.ascore(**kwargs)
        return extract_metric_value(raw_result)
    except Exception as e:
        print(f"[WARN] {type(metric).__name__} failed: {e}")
        return None



# Context text formatting

def stringify_graph_element(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x.strip()

    if isinstance(x, (list, tuple)):
        if len(x) == 3:
            s, p, o = x
            return f"{s} {p} {o}"
        return " | ".join(str(v) for v in x)

    if isinstance(x, dict):
        subject = (
            x.get("subject")
            or x.get("source")
            or x.get("src")
            or x.get("from")
            or x.get("head")
        )
        predicate = (
            x.get("predicate")
            or x.get("relation")
            or x.get("edge")
            or x.get("label")
            or x.get("type")
        )
        obj = (
            x.get("object")
            or x.get("target")
            or x.get("dst")
            or x.get("to")
            or x.get("tail")
        )

        if subject is not None and predicate is not None and obj is not None:
            return f"{subject} {predicate} {obj}"

        return json.dumps(x, ensure_ascii=False, sort_keys=True)

    return str(x)


def build_text_contexts(elements: List[Any]) -> List[str]:
    return [
        c for c in (stringify_graph_element(e) for e in (elements or []))
        if c and c.strip()
    ]


def normalize_faithfulness_contexts(items: Any) -> List[str]:
    if items is None:
        return []

    if isinstance(items, str):
        value = items.strip()
        return [value] if value else []

    if isinstance(items, list):
        cleaned = []
        for item in items:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    text = str(items).strip()
    return [text] if text else []

# Canonical element parsing
ENTITY_RE = re.compile(
    r"^(?P<name>.+?)\s*\(type:\s*(?P<etype>[^)]+)\)\s*$",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_outer_parens(text: str) -> str:
    """
    Removes one or more layers of balanced outer parentheses:
      "(A)" -> "A"
      "((A))" -> "A"
    but preserves inner structure.
    """
    s = text.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def canonicalize_name(text: str) -> str:
    s = normalize_whitespace(text)
    s = s.casefold()

    s = s.replace("&", " and ")
    s = s.replace("/", " / ")
    s = s.replace("’", "'")
    s = s.replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')

    s = s.strip(" \t\n\r.,;:")

    s = normalize_whitespace(s)
    return s


def canonicalize_type(text: str) -> str:
    s = normalize_whitespace(text).casefold()
    return s


def canonicalize_relation_type(text: str) -> str:
    s = normalize_whitespace(text).casefold()
    return s


def canonicalize_relation_label(text: str) -> str:
    s = normalize_whitespace(text).casefold()
    s = s.strip("[]")
    return s


def normalize_element_raw(x: Any) -> str:
    """
    Converts arbitrary item to a string for parsing.
    """
    if x is None:
        return ""

    if isinstance(x, str):
        return normalize_whitespace(x)

    if isinstance(x, (int, float, bool)):
        return str(x).strip()

    if isinstance(x, (list, tuple)):
        if len(x) == 3:
            return normalize_whitespace(f"{x[0]} {x[1]} {x[2]}")
        return normalize_whitespace(" | ".join(str(v) for v in x))

    if isinstance(x, dict):
        return stringify_graph_element(x)

    return normalize_whitespace(str(x))


def find_top_level_relation(text: str) -> Optional[Tuple[int, int, str]]:
    """
    Finds the first top-level relation token:
      ... <RelationType> ...
    where RelationType is a token ending with 'Relationship'

    Returns (start_idx, end_idx, relation_type) or None.
    """
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue

        if depth == 0:
            m = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\b", text[i:])
            if m:
                token = m.group(1)
                if token.lower().endswith("relationship"):
                    start = i + m.start(1)
                    end = i + m.end(1)
                    return start, end, token
                i += max(1, m.end(0))
                continue

        i += 1

    return None


def parse_optional_relation_label(text: str) -> Tuple[Optional[str], str]:
    """
    Parses:
      "[label] rest"
    or returns (None, original)
    """
    s = text.strip()
    if not s.startswith("["):
        return None, s

    depth = 0
    for i, ch in enumerate(s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                label = s[1:i].strip()
                rest = s[i + 1 :].strip()
                return label, rest

    return None, s


def parse_entity(text: str) -> Optional[Tuple[str, str]]:
    s = strip_outer_parens(text)
    m = ENTITY_RE.match(s)
    if not m:
        return None

    name = canonicalize_name(m.group("name"))
    etype = canonicalize_type(m.group("etype"))
    return name, etype


def parse_element(text: str) -> Optional[Tuple]:
    """
    Canonical recursive parser.

    Output shapes:
      ("entity", name, type)
      ("relation", left_struct, relation_type, relation_label_or_None, right_struct)

    Supports:
      entity
      entity REL entity
      entity REL [label] entity
      (entity REL entity) REL entity
    """
    raw = normalize_element_raw(text)
    if not raw:
        return None

    raw = strip_outer_parens(raw)

    entity = parse_entity(raw)
    if entity is not None:
        return ("entity", entity[0], entity[1])

    rel = find_top_level_relation(raw)
    if rel is None:
        return ("text", canonicalize_name(raw))

    start, end, rel_type = rel
    left_text = raw[:start].strip()
    right_text = raw[end:].strip()

    label, right_text = parse_optional_relation_label(right_text)

    left_struct = parse_element(left_text)
    right_struct = parse_element(right_text)

    if left_struct is None or right_struct is None:
        return (
            "relation_text",
            canonicalize_name(left_text),
            canonicalize_relation_type(rel_type),
            canonicalize_relation_label(label) if label else None,
            canonicalize_name(right_text),
        )

    return (
        "relation",
        left_struct,
        canonicalize_relation_type(rel_type),
        canonicalize_relation_label(label) if label else None,
        right_struct,
    )


def canonicalize_element(item: Any) -> Optional[Tuple]:
    raw = normalize_element_raw(item)
    if not raw:
        return None
    return parse_element(raw)


def canonicalize_element_list(items: Any) -> List[Tuple]:
    if not items:
        return []

    out = []
    seen = set()

    for item in items:
        canonical = canonicalize_element(item)
        if canonical is None:
            continue
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)

    return out


def canonical_element_to_string(el: Tuple) -> str:
    kind = el[0]

    if kind == "entity":
        _, name, etype = el
        return f"{name} (type:{etype})"

    if kind == "text":
        return el[1]

    if kind == "relation_text":
        _, left, rel_type, label, right = el
        if label:
            return f"{left} {rel_type} [{label}] {right}"
        return f"{left} {rel_type} {right}"

    if kind == "relation":
        _, left, rel_type, label, right = el
        left_s = canonical_element_to_string(left)
        right_s = canonical_element_to_string(right)
        if left[0] == "relation":
            left_s = f"({left_s})"
        if right[0] == "relation":
            right_s = f"({right_s})"
        if label:
            return f"{left_s} {rel_type} [{label}] {right_s}"
        return f"{left_s} {rel_type} {right_s}"

    return str(el)



# Custom element metrics
def overlap_count(a: List[Tuple], b: List[Tuple]) -> int:
    return len(set(a).intersection(set(b)))


def calculate_elements_used_pct(
    elements_included_in_prompt: List[Any],
    elements_used: List[Any],
) -> Optional[float]:
    prompt_elements = canonicalize_element_list(elements_included_in_prompt)
    used_elements = canonicalize_element_list(elements_used)

    if not prompt_elements:
        return None

    return overlap_count(prompt_elements, used_elements) / len(set(prompt_elements))


def calculate_precision(
    retrieved_elements: List[Any],
    reference_elements: List[Any],
) -> Optional[float]:
    retrieved = canonicalize_element_list(retrieved_elements)
    reference = canonicalize_element_list(reference_elements)

    if not retrieved:
        return None

    return overlap_count(retrieved, reference) / len(set(retrieved))


def calculate_recall(
    retrieved_elements: List[Any],
    reference_elements: List[Any],
) -> Optional[float]:
    retrieved = canonicalize_element_list(retrieved_elements)
    reference = canonicalize_element_list(reference_elements)

    if not reference:
        return None

    return overlap_count(retrieved, reference) / len(set(reference))


def calculate_f1(
    precision: Optional[float],
    recall: Optional[float],
) -> float:
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



# Main
async def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    records = load_records(INPUT_FILE)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a top-level list of records.")

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    evaluator_llm = llm_factory(
        LLM_MODEL,
        client=client,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    evaluator_embeddings = OpenAIEmbeddings(
        client=client,
        model=EMBEDDING_MODEL,
    )

    response_relevancy = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    factual_correctness = FactualCorrectness(llm=evaluator_llm)
    semantic_similarity = SemanticSimilarity(embeddings=evaluator_embeddings)
    faithfulness = Faithfulness(llm=evaluator_llm)

    rows = []

    for i, rec in enumerate(records, start=1):
        question = rec.get("Question", "")
        response = rec.get("Response", "")
        expected_answer = rec.get("Expected Answer", "")
        classification = rec.get("Classification", "")

        elements_included_in_prompt = rec.get("Elements Included In Prompt", [])
        elements_used = rec.get("Elements Used", [])
        expected_elements = rec.get("Expected Elements", [])

        retrieved_contexts_derived = build_text_contexts(elements_included_in_prompt)

        faithfulness_contexts = normalize_faithfulness_contexts(
            rec.get("Faithfulness Contexts", [])
        )

        retrieved_contexts_for_faithfulness = (
            faithfulness_contexts if faithfulness_contexts else retrieved_contexts_derived
        )

        canonical_prompt_elements = canonicalize_element_list(elements_included_in_prompt)
        canonical_used_elements = canonicalize_element_list(elements_used)
        canonical_expected_elements = canonicalize_element_list(expected_elements)

        canonical_prompt_strings = [canonical_element_to_string(x) for x in canonical_prompt_elements]
        canonical_used_strings = [canonical_element_to_string(x) for x in canonical_used_elements]
        canonical_expected_strings = [canonical_element_to_string(x) for x in canonical_expected_elements]

        elements_used_pct = calculate_elements_used_pct(
            elements_included_in_prompt=elements_included_in_prompt,
            elements_used=elements_used,
        )

        element_precision = calculate_precision(
            retrieved_elements=elements_used,
            reference_elements=expected_elements,
        )

        element_recall = calculate_recall(
            retrieved_elements=elements_used,
            reference_elements=expected_elements,
        )

        element_f1 = calculate_f1(element_precision, element_recall)

        response_relevancy_score = await safe_score_collections(
            response_relevancy,
            user_input=question,
            response=response,
        )

        factual_correctness_score = await safe_score_collections(
            factual_correctness,
            response=response,
            reference=expected_answer,
        )

        semantic_similarity_score = await safe_score_collections(
            semantic_similarity,
            response=response,
            reference=expected_answer,
        )

        faithfulness_score = await safe_score_collections(
            faithfulness,
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts_for_faithfulness,
        )

        rows.append(
            {
                "classification": classification,
                "question": question,
                "response": response,
                "expected_answer": expected_answer,
                "elements_included_in_prompt": elements_included_in_prompt,
                "elements_used": elements_used,
                "expected_elements": expected_elements,
                "faithfulness_contexts": faithfulness_contexts,
                "retrieved_contexts_derived": retrieved_contexts_derived,
                "retrieved_contexts_used_for_faithfulness": retrieved_contexts_for_faithfulness,

                # Canonical/debug columns
                "canonical_prompt_elements": canonical_prompt_strings,
                "canonical_used_elements": canonical_used_strings,
                "canonical_expected_elements": canonical_expected_strings,

                # Custom element metrics
                "elements_used_pct": elements_used_pct,
                "element_precision": element_precision,
                "element_recall": element_recall,
                "element_f1": element_f1,

                # Ragas text metrics
                "response_relevancy": response_relevancy_score,
                "factual_correctness": factual_correctness_score,
                "semantic_similarity": semantic_similarity_score,
                "faithfulness": faithfulness_score,
            }
        )

        print(f"[{i}/{len(records)}] processed")

    df = pd.DataFrame(rows)

    print("\nPer-example results:")
    print(df)

    summary_cols = [
        "elements_used_pct",
        "response_relevancy",
        "factual_correctness",
        "semantic_similarity",
        "faithfulness",
        "element_precision",
        "element_recall",
        "element_f1",
    ]

    print("\nAverages:")
    for col in summary_cols:
        print(f"{col}: {safe_mean(df[col]):.4f}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())