from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Any

from openai import OpenAI
from pydantic import BaseModel, Field


# Defaults (can be overridden by CLI args)

FILE_PATH = ""
QUESTIONS_JSON_PATH = ""
DEFAULT_OUTDIR = ""
DEFAULT_OUTPUT_JSON = ""
LLM_IO_SUBDIR = ""
QUESTION_LIMIT = 0

# Defaults relating to LLM (can be overridden by CLI args)
MODEL = "gpt-5.2"
REASONING_EFFORT = "none"
TEMPERATURE = 0.2


# XML helpers
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XSI_TYPE_ATTR = f"{{{XSI_NS}}}type"


@dataclass(frozen=True)
class EntityRecord:
    element_id: str
    name: str | None
    xsi_type: str
    type_name: str


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str | None
    relation_name: str | None
    source_id: str | None
    relation_type: str
    target_id: str | None


class TypeSelection(BaseModel):
    selected_entity_types: list[str] = Field(default_factory=list)


class EntitySelection(BaseModel):
    relevant_entities: list[str] = Field(default_factory=list)


class RelationTypeSelection(BaseModel):
    selected_relationship_types: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    can_answer: bool
    response: str = ""
    elements_used: list[str] = Field(default_factory=list)


# LLM artifact logging
@dataclass
class LLMArtifactLogger:
    out_dir: Path
    subdir_name: str = LLM_IO_SUBDIR
    counter: int = 0
    saved_files: list[dict[str, str]] = field(default_factory=list)

    @property
    def llm_io_dir(self) -> Path:
        return self.out_dir / self.subdir_name

    def next_index(self) -> int:
        self.counter += 1
        return self.counter

    def register(self, prompt_name: str, prompt_file: str, output_file: str) -> None:
        self.saved_files.append(
            {
                "call_index": f"{self.counter:02d}",
                "prompt_name": prompt_name,
                "prompt_file": f"{self.subdir_name}/{prompt_file}",
                "output_file": f"{self.subdir_name}/{output_file}",
            }
        )


# General helpers

def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "item"


def render_combined_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "=== SYSTEM PROMPT ===\n"
        f"{system_prompt}\n\n"
        "=== USER PROMPT ===\n"
        f"{user_prompt}\n"
    )


def model_output_as_text(response: Any, parsed: BaseModel | None) -> str:
    raw_output_text = getattr(response, "output_text", None)

    if raw_output_text:
        return str(raw_output_text)

    if parsed is not None:
        try:
            return json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            return str(parsed)

    return "(No output text returned by model.)"


def save_llm_call_artifacts(
    logger: LLMArtifactLogger,
    prompt_name: str,
    system_prompt: str,
    user_prompt: str,
    response: Any,
    parsed: BaseModel | None,
) -> None:
    logger.llm_io_dir.mkdir(parents=True, exist_ok=True)
    call_index = logger.next_index()
    safe_name = sanitize_filename(prompt_name)

    prompt_filename = f"{call_index:02d}_llm_prompt_{safe_name}.txt"
    output_filename = f"{call_index:02d}_llm_output_{safe_name}.txt"

    prompt_path = logger.llm_io_dir / prompt_filename
    output_path = logger.llm_io_dir / output_filename

    prompt_path.write_text(render_combined_prompt(system_prompt, user_prompt), encoding="utf-8")
    output_path.write_text(model_output_as_text(response, parsed), encoding="utf-8")

    logger.register(prompt_name=prompt_name, prompt_file=prompt_filename, output_file=output_filename)


def _write_list_txt(out_dir: Path, filename: str, data: list[Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_txt(out_dir: Path, filename: str, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(text, encoding="utf-8")


def _write_json(out_path: Path, data: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Parsing / sanitizing helpers

def short_type_name(type_value: str | None) -> str:
    if not type_value:
        return ""
    return type_value.split(":", 1)[1] if ":" in type_value else type_value


def parse_archimate_model(
    file_path: Path,
) -> tuple[list[EntityRecord], list[RelationRecord], dict[str, EntityRecord], dict[str, RelationRecord]]:
    entities: list[EntityRecord] = []
    relations: list[RelationRecord] = []
    id_to_entity: dict[str, EntityRecord] = {}
    id_to_relation: dict[str, RelationRecord] = {}

    context = ET.iterparse(str(file_path), events=("start", "end"))

    in_relations_folder = False
    depth = 0

    for event, elem in context:
        if event == "start":
            if in_relations_folder:
                depth += 1
            elif elem.tag == "folder" and elem.attrib.get("type") == "relations":
                in_relations_folder = True
                depth = 1
            continue

        if in_relations_folder:
            if elem.tag == "element":
                relation = RelationRecord(
                    relation_id=elem.attrib.get("id") or elem.attrib.get("identifier"),
                    relation_name=elem.attrib.get("name"),
                    source_id=elem.attrib.get("source"),
                    relation_type=short_type_name(elem.attrib.get(XSI_TYPE_ATTR) or elem.attrib.get("xsi:type")),
                    target_id=elem.attrib.get("target"),
                )
                relations.append(relation)
                if relation.relation_id:
                    id_to_relation[relation.relation_id] = relation

            depth -= 1
            if depth == 0:
                in_relations_folder = False

            elem.clear()
            continue

        if elem.tag == "element":
            elem_id = elem.attrib.get("id") or elem.attrib.get("identifier")
            name = elem.attrib.get("name")
            xsi_type = elem.attrib.get(XSI_TYPE_ATTR) or elem.attrib.get("xsi:type")

            if elem_id and xsi_type:
                entity = EntityRecord(
                    element_id=elem_id,
                    name=name,
                    xsi_type=xsi_type,
                    type_name=short_type_name(xsi_type),
                )
                entities.append(entity)
                id_to_entity[elem_id] = entity

        elem.clear()

    return entities, relations, id_to_entity, id_to_relation


def unique_entity_types(entities: list[EntityRecord]) -> list[str]:
    return sorted({e.type_name for e in entities if e.type_name})


def unique_relation_types(relations: list[RelationRecord]) -> list[str]:
    return sorted({r.relation_type for r in relations if r.relation_type})


def sanitize_values(raw_values: list[str], available_values: list[str]) -> list[str]:
    available_map = {v.casefold(): v for v in available_values}
    cleaned: list[str] = []

    for raw in raw_values:
        if not raw:
            continue
        candidate = short_type_name(raw.strip())
        actual = available_map.get(candidate.casefold())
        if actual and actual not in cleaned:
            cleaned.append(actual)

    return cleaned


def sanitize_exact_strings(raw_values: list[str], available_values: list[str]) -> list[str]:
    available_map = {v.strip().casefold(): v for v in available_values}
    cleaned: list[str] = []

    for raw in raw_values:
        if not raw:
            continue
        actual = available_map.get(raw.strip().casefold())
        if actual and actual not in cleaned:
            cleaned.append(actual)

    return cleaned


def sanitize_types(raw_types: list[str], available_types: list[str]) -> list[str]:
    return sanitize_values(raw_types, available_types)


def sanitize_relation_types(raw_types: list[str], available_types: list[str]) -> list[str]:
    return sanitize_values(raw_types, available_types)


def entities_by_selected_types(
    entities: list[EntityRecord],
    selected_types: list[str],
) -> dict[str, list[EntityRecord]]:
    wanted = set(selected_types)
    grouped: dict[str, list[EntityRecord]] = {t: [] for t in selected_types}

    for entity in entities:
        if entity.type_name in wanted and entity.name:
            grouped.setdefault(entity.type_name, []).append(entity)

    for type_name in grouped:
        grouped[type_name].sort(key=lambda e: ((e.name or "").casefold(), e.element_id))

    return grouped


def format_entity_label(name: str, type_name: str) -> str:
    return f"{name} (type: {type_name})"


def candidate_entity_labels(grouped_entities: dict[str, list[EntityRecord]]) -> list[str]:
    labels = {
        format_entity_label(entity.name, entity.type_name)
        for entity_list in grouped_entities.values()
        for entity in entity_list
        if entity.name and entity.type_name
    }
    return sorted(labels, key=lambda x: x.casefold())


def sanitize_entity_labels_to_names(
    raw_values: list[str],
    grouped_entities: dict[str, list[EntityRecord]],
) -> list[str]:
    label_to_name: dict[str, str] = {}
    for entity_list in grouped_entities.values():
        for entity in entity_list:
            if entity.name and entity.type_name:
                label = format_entity_label(entity.name, entity.type_name)
                label_to_name[label.casefold()] = entity.name

    cleaned: list[str] = []
    for raw in raw_values:
        if not raw:
            continue
        actual_name = label_to_name.get(raw.strip().casefold())
        if actual_name and actual_name not in cleaned:
            cleaned.append(actual_name)

    return cleaned


def retrieved_entities_from_names(
    entities: list[EntityRecord],
    selected_names: list[str],
) -> list[dict[str, str]]:
    wanted = set(selected_names)
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str | None, str]] = set()

    for entity in entities:
        if entity.name in wanted:
            key = (entity.element_id, entity.name, entity.type_name)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "id": entity.element_id,
                    "name": entity.name,
                    "type": entity.type_name,
                }
            )

    results.sort(key=lambda x: ((x["name"] or "").casefold(), x["type"].casefold(), x["id"]))
    return results


def retrieved_entity_labels(retrieved_entities: list[dict[str, str]]) -> list[str]:
    labels = {
        format_entity_label(e["name"], e["type"])
        for e in retrieved_entities
        if e.get("name") and e.get("type")
    }
    return sorted(labels, key=lambda x: x.casefold())


def resolve_reference_label(
    ref_id: str | None,
    id_to_entity: dict[str, EntityRecord],
    id_to_relation: dict[str, RelationRecord],
    memo: dict[str, str] | None = None,
    visiting: set[str] | None = None,
) -> str:
    if not ref_id:
        return "(unknown:None)"

    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()

    if ref_id in memo:
        return memo[ref_id]

    entity = id_to_entity.get(ref_id)
    if entity is not None:
        if entity.name and entity.type_name:
            label = format_entity_label(entity.name, entity.type_name)
        elif entity.name:
            label = entity.name
        else:
            label = f"(unnamed:{ref_id})"
        memo[ref_id] = label
        return label

    relation = id_to_relation.get(ref_id)
    if relation is None:
        label = f"(unknown:{ref_id})"
        memo[ref_id] = label
        return label

    if ref_id in visiting:
        return f"(cyclic_relation:{ref_id})"

    visiting.add(ref_id)

    source_label = resolve_reference_label(relation.source_id, id_to_entity, id_to_relation, memo, visiting)
    target_label = resolve_reference_label(relation.target_id, id_to_entity, id_to_relation, memo, visiting)
    relation_name = (relation.relation_name or "").strip()

    if relation_name:
        label = f"({source_label} {relation.relation_type} [{relation_name}] {target_label})"
    else:
        label = f"({source_label} {relation.relation_type} {target_label})"

    visiting.remove(ref_id)
    memo[ref_id] = label
    return label


def reference_touches_selected_name(
    ref_id: str | None,
    wanted_names: set[str],
    id_to_entity: dict[str, EntityRecord],
    id_to_relation: dict[str, RelationRecord],
    memo: dict[str, bool] | None = None,
    visiting: set[str] | None = None,
) -> bool:
    if not ref_id:
        return False

    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()

    if ref_id in memo:
        return memo[ref_id]

    entity = id_to_entity.get(ref_id)
    if entity is not None:
        result = entity.name in wanted_names if entity.name else False
        memo[ref_id] = result
        return result

    relation = id_to_relation.get(ref_id)
    if relation is None:
        memo[ref_id] = False
        return False

    if ref_id in visiting:
        return False

    visiting.add(ref_id)
    result = (
        reference_touches_selected_name(relation.source_id, wanted_names, id_to_entity, id_to_relation, memo, visiting)
        or reference_touches_selected_name(relation.target_id, wanted_names, id_to_entity, id_to_relation, memo, visiting)
    )
    visiting.remove(ref_id)
    memo[ref_id] = result
    return result


def retrieve_relations_for_names_and_types(
    relations: list[RelationRecord],
    id_to_entity: dict[str, EntityRecord],
    id_to_relation: dict[str, RelationRecord],
    selected_names: list[str],
    selected_relation_types: list[str],
) -> list[dict[str, Any]]:
    wanted_names = set(selected_names)
    wanted_relation_types = set(selected_relation_types)

    results: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str, str | None]] = set()

    label_memo: dict[str, str] = {}
    touches_memo: dict[str, bool] = {}

    for rel in relations:
        if wanted_relation_types and rel.relation_type not in wanted_relation_types:
            continue

        source_label = resolve_reference_label(
            rel.source_id,
            id_to_entity=id_to_entity,
            id_to_relation=id_to_relation,
            memo=label_memo,
        )
        target_label = resolve_reference_label(
            rel.target_id,
            id_to_entity=id_to_entity,
            id_to_relation=id_to_relation,
            memo=label_memo,
        )

        if wanted_names:
            touches_selected_name = (
                reference_touches_selected_name(
                    rel.source_id,
                    wanted_names=wanted_names,
                    id_to_entity=id_to_entity,
                    id_to_relation=id_to_relation,
                    memo=touches_memo,
                )
                or reference_touches_selected_name(
                    rel.target_id,
                    wanted_names=wanted_names,
                    id_to_entity=id_to_entity,
                    id_to_relation=id_to_relation,
                    memo=touches_memo,
                )
            )
            if not touches_selected_name:
                continue

        key = (rel.source_id, rel.relation_name, rel.relation_type, rel.target_id)
        if key in seen:
            continue
        seen.add(key)

        results.append(
            {
                "relation_id": rel.relation_id,
                "relation_name": rel.relation_name,
                "source_id": rel.source_id,
                "source_name": source_label,
                "relation_type": rel.relation_type,
                "target_id": rel.target_id,
                "target_name": target_label,
            }
        )

    results.sort(
        key=lambda x: (
            (x["source_name"] or "").casefold(),
            (x["relation_type"] or "").casefold(),
            (x["relation_name"] or "").casefold(),
            (x["target_name"] or "").casefold(),
        )
    )
    return results


def should_retry_relation_answer(answer: FinalAnswer) -> bool:
    return (not answer.can_answer) or (not answer.response.strip())


def relation_triplets(relations: list[dict[str, Any]]) -> list[str]:
    triplets: list[str] = []
    for rel in relations:
        source_name = rel.get("source_name") or ""
        relation_type = rel.get("relation_type") or ""
        relation_name = (rel.get("relation_name") or "").strip()
        target_name = rel.get("target_name") or ""

        if relation_name:
            triplets.append(f"{source_name} {relation_type} [{relation_name}] {target_name}")
        else:
            triplets.append(f"{source_name} {relation_type} {target_name}")
    return triplets


def _flatten_grouped_candidates(grouped: dict[str, list[EntityRecord]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for type_name in sorted(grouped.keys(), key=lambda x: x.casefold()):
        for e in grouped[type_name]:
            flat.append(
                {
                    "type": type_name,
                    "name": e.name,
                    "label": format_entity_label(e.name, type_name) if e.name else None,
                    "id": e.element_id,
                }
            )
    flat.sort(key=lambda x: ((x["type"] or "").casefold(), (x["name"] or "").casefold(), x["id"]))
    return flat


# =========================
# Token usage helpers
# =========================

def extract_usage_summary(response: Any, prompt_name: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None) if usage is not None else None
    output_details = getattr(usage, "output_tokens_details", None) if usage is not None else None

    return {
        "prompt": prompt_name,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "cached_input_tokens": getattr(input_details, "cached_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def render_prompt_usage_report(prompt_usages: list[dict[str, Any]], question: str) -> str:
    if not prompt_usages:
        return f"Question: {question}\n\nNo LLM calls were made."

    total_input = sum(item.get("input_tokens", 0) or 0 for item in prompt_usages)
    total_cached = sum(item.get("cached_input_tokens", 0) or 0 for item in prompt_usages)
    total_output = sum(item.get("output_tokens", 0) or 0 for item in prompt_usages)
    total_reasoning = sum(item.get("reasoning_tokens", 0) or 0 for item in prompt_usages)
    total_tokens = sum(item.get("total_tokens", 0) or 0 for item in prompt_usages)

    lines: list[str] = [
        f"Question: {question}",
        "",
        "LLM prompt token breakdown",
        "==========================",
        "",
    ]

    for item in prompt_usages:
        lines.extend(
            [
                f"Prompt: {item['prompt']}",
                f"  input_tokens: {item['input_tokens']}",
                f"  cached_input_tokens: {item['cached_input_tokens']}",
                f"  output_tokens: {item['output_tokens']}",
                f"  reasoning_tokens: {item['reasoning_tokens']}",
                f"  total_tokens: {item['total_tokens']}",
                "",
            ]
        )

    lines.extend(
        [
            "Question totals",
            "===============",
            f"input_tokens: {total_input}",
            f"cached_input_tokens: {total_cached}",
            f"output_tokens: {total_output}",
            f"reasoning_tokens: {total_reasoning}",
            f"total_tokens: {total_tokens}",
        ]
    )
    return "\n".join(lines)


def render_all_questions_usage_report(all_question_usages: list[dict[str, Any]]) -> str:
    if not all_question_usages:
        return "No question-level LLM token usage data was collected."

    grand_input = 0
    grand_cached = 0
    grand_output = 0
    grand_reasoning = 0
    grand_total = 0

    lines: list[str] = [
        "All questions LLM prompt token usage",
        "====================================",
        "",
    ]

    for idx, item in enumerate(all_question_usages, start=1):
        question = item.get("question", "")
        prompt_usages = item.get("prompt_usages", []) or []

        q_input = sum(p.get("input_tokens", 0) or 0 for p in prompt_usages)
        q_cached = sum(p.get("cached_input_tokens", 0) or 0 for p in prompt_usages)
        q_output = sum(p.get("output_tokens", 0) or 0 for p in prompt_usages)
        q_reasoning = sum(p.get("reasoning_tokens", 0) or 0 for p in prompt_usages)
        q_total = sum(p.get("total_tokens", 0) or 0 for p in prompt_usages)

        grand_input += q_input
        grand_cached += q_cached
        grand_output += q_output
        grand_reasoning += q_reasoning
        grand_total += q_total

        lines.extend(
            [
                f"Question {idx}: {question}",
                "-" * 80,
            ]
        )

        if prompt_usages:
            for prompt_item in prompt_usages:
                lines.extend(
                    [
                        f"Prompt: {prompt_item['prompt']}",
                        f"  input_tokens: {prompt_item['input_tokens']}",
                        f"  cached_input_tokens: {prompt_item['cached_input_tokens']}",
                        f"  output_tokens: {prompt_item['output_tokens']}",
                        f"  reasoning_tokens: {prompt_item['reasoning_tokens']}",
                        f"  total_tokens: {prompt_item['total_tokens']}",
                        "",
                    ]
                )
        else:
            lines.append("No LLM calls were made.")
            lines.append("")

        lines.extend(
            [
                "Question totals",
                f"  input_tokens: {q_input}",
                f"  cached_input_tokens: {q_cached}",
                f"  output_tokens: {q_output}",
                f"  reasoning_tokens: {q_reasoning}",
                f"  total_tokens: {q_total}",
                "",
                "",
            ]
        )

    lines.extend(
        [
            "Grand totals",
            "============",
            f"input_tokens: {grand_input}",
            f"cached_input_tokens: {grand_cached}",
            f"output_tokens: {grand_output}",
            f"reasoning_tokens: {grand_reasoning}",
            f"total_tokens: {grand_total}",
        ]
    )

    return "\n".join(lines)


def render_llm_io_index(saved_files: list[dict[str, str]], subdir_name: str) -> str:
    if not saved_files:
        return "No LLM prompt/output files were saved."

    lines = [
        "LLM prompt/output files",
        "=======================",
        "",
        f"Subfolder: {subdir_name}",
        "",
    ]
    for item in saved_files:
        lines.extend(
            [
                f"Call {item['call_index']}: {item['prompt_name']}",
                f"  Prompt file: {item['prompt_file']}",
                f"  Output file: {item['output_file']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


# =========================
# OpenAI helpers
# =========================

def call_structured_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    schema: Type[BaseModel],
    model: str,
    prompt_name: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[BaseModel, dict[str, Any]]:
    response = client.responses.parse(
        model=model,
        reasoning={"effort": REASONING_EFFORT},
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=schema,
    )

    parsed = response.output_parsed

    if llm_logger is not None:
        try:
            save_llm_call_artifacts(
                logger=llm_logger,
                prompt_name=prompt_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                parsed=parsed,
            )
        except OSError as exc:
            print(f"Warning: could not write LLM prompt/output artifacts: {exc}", file=sys.stderr)

    if parsed is None:
        raise RuntimeError(f"Model returned no structured output. Raw output: {response.output_text}")

    usage_summary = extract_usage_summary(response, prompt_name)
    return parsed, usage_summary


def select_relevant_entity_types(
    client: OpenAI,
    question: str,
    all_entity_types: list[str],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[list[str], dict[str, Any]]:
    system_prompt = (
        "You are selecting ArchiMate ENTITY TYPES relevant for answering a user's question.\n"
        "Select all entity types that could help answer the question by considering what types can be used "
        "to represent the entities in the question and the entities in the expected answer.\n"
        "Return only types from the provided list.\n"
        "Do not invent types.\n"
        "Consider that the query may be written by a user who is not familiar with ArchiMate terminology.\n"
        "If nothing is relevant, or if the question relates mainly to relationships/flows/relations, return an empty list."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Available entity types:\n{json.dumps(all_entity_types, ensure_ascii=False, indent=2)}"
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=TypeSelection,
        model=model,
        prompt_name="select_relevant_entity_types",
        llm_logger=llm_logger,
    )
    return sanitize_types(parsed.selected_entity_types, all_entity_types), usage


def select_relevant_entities(
    client: OpenAI,
    question: str,
    grouped_entities: dict[str, list[EntityRecord]],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[list[str], dict[str, Any]]:
    all_candidate_labels = candidate_entity_labels(grouped_entities)

    system_prompt = (
        "You are selecting ArchiMate entities that can support answering a user's question.\n"
        "Select entities needed to represent the question as well as possible answers.\n"
        "Also select entities that may be indirectly relevant through typical ArchiMate relationships, "
        "even if their names are not an obvious textual match.\n"
        "You will receive candidate entities in the exact form 'Name (type: Type)'.\n"
        "Return only exact candidate strings from the provided list.\n"
        "Consider that the query may be ambiguous or too general.\n"
        "Do not invent names or types.\n"
        "Do not assume the relationships between entities.\n"
        "If none are relevant, return an empty list."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Candidate entities:\n{json.dumps(all_candidate_labels, ensure_ascii=False, indent=2)}"
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=EntitySelection,
        model=model,
        prompt_name="select_relevant_entities",
        llm_logger=llm_logger,
    )
    return sanitize_entity_labels_to_names(parsed.relevant_entities, grouped_entities), usage


def select_relevant_relationship_types(
    client: OpenAI,
    question: str,
    all_relationship_types: list[str],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[list[str], dict[str, Any]]:
    system_prompt = (
        "You are selecting ArchiMate RELATIONSHIP TYPES relevant for answering a user's question.\n"
        "Select all relationship types that could help answer the question.\n"
        "Consider that the answer may require information from multiple combinations of relationships.\n"
        "Return only relationship types from the provided list.\n"
        "Do not invent relationship types.\n"
        "Consider that the query may be ambiguous or too general.\n"
        "If nothing is relevant, return an empty list."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Available relationship types:\n{json.dumps(all_relationship_types, ensure_ascii=False, indent=2)}"
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=RelationTypeSelection,
        model=model,
        prompt_name="select_relevant_relationship_types",
        llm_logger=llm_logger,
    )
    return sanitize_relation_types(parsed.selected_relationship_types, all_relationship_types), usage


def answer_from_entities_only(
    client: OpenAI,
    question: str,
    retrieved_entities: list[dict[str, str]],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[FinalAnswer, dict[str, Any], list[str]]:
    available_entity_labels = retrieved_entity_labels(retrieved_entities)

    system_prompt = (
        "You are a business analyst answering questions based only on entities from an ArchiMate model.\n"
        "Use only the provided entities.\n"
        "Some of the entities may not be relevant.\n"
        "Relationships between entities are not provided, so you must not assume any.\n"
        "Provide the answer in the form of single paragraph that includes the relavant entity names and their types in the exact form 'Name (type: Type)'.\n"
        "If relationships are needed, set can_answer=false.\n"
        "Do not use external knowledge.\n"
        "Each entity is provided in the exact form 'Name (type: Type)'.\n"
        "If can_answer=true, elements_used must contain only exact entity strings from the provided list that directly support the answer.\n"
        "Preserve exact string formatting.\n"
        "If can_answer=false, set elements_used to an empty list."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Retrieved entities:\n{json.dumps(available_entity_labels, ensure_ascii=False, indent=2)}\n"
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=FinalAnswer,
        model=model,
        prompt_name="answer_from_entities_only",
        llm_logger=llm_logger,
    )

    final_answer = FinalAnswer(
        can_answer=parsed.can_answer,
        response=parsed.response,
        elements_used=sanitize_exact_strings(parsed.elements_used, available_entity_labels),
    )
    return final_answer, usage, available_entity_labels


def answer_from_relations(
    client: OpenAI,
    question: str,
    retrieved_relations: list[dict[str, Any]],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[FinalAnswer, dict[str, Any], list[str]]:
    relation_strings = relation_triplets(retrieved_relations)

    system_prompt = (
        "You are a business analyst answering questions based on relations from an ArchiMate model.\n"
        "Use only the provided relations.\n"
        "Some of the relations may not be relevant.\n"
        "Do not assume any relationships unless explicitly given.\n"
        "Do not use external knowledge.\n"
        "Relationship names are meaningful. If a relation string includes a bracketed segment like [CRM data],"
        "treat that as the explicit relationship name and use it when relevant.\n"
        "If the question cannot be answered from the provided data, set can_answer=false.\n"
        "If can_answer=true, elements_used must contain only exact relation strings from the provided list that directly support the answer.\n"
        "Provide the answer in the form of single paragraph that includes the relavant entity names and their types in the exact form 'Name (type: Type)'.\n"
        "Each relation string will be exactly one of these forms:\n"
        "  - 'source_label relation_type target_label'\n"
        "  - 'source_label relation_type [relation_name] target_label'\n"
        "source_label and target_label may also be recursively nested relation expressions.\n"
        "Preserve the exact string when returning elements_used.\n"
        "If can_answer=false, set elements_used to an empty list."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Retrieved relations:\n{json.dumps(relation_strings, ensure_ascii=False, indent=2)}\n"
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=FinalAnswer,
        model=model,
        prompt_name="answer_from_relations",
        llm_logger=llm_logger,
    )

    final_answer = FinalAnswer(
        can_answer=parsed.can_answer,
        response=parsed.response,
        elements_used=sanitize_exact_strings(parsed.elements_used, relation_strings),
    )
    return final_answer, usage, relation_strings


# =========================
# Question processing
# =========================

@dataclass
class QuestionRunResult:
    question: str
    response: str
    elements_used: list[str]
    elements_included_in_prompt: list[str]
    prompt_usages: list[dict[str, Any]]


def process_single_question(
    client: OpenAI,
    question: str,
    model: str,
    question_out_dir: Path,
    entities: list[EntityRecord],
    relations: list[RelationRecord],
    id_to_entity: dict[str, EntityRecord],
    id_to_relation: dict[str, RelationRecord],
    all_entity_types: list[str],
    all_relationship_types: list[str],
) -> QuestionRunResult:
    llm_logger = LLMArtifactLogger(out_dir=question_out_dir)
    prompt_usages: list[dict[str, Any]] = []

    _write_list_txt(question_out_dir, "01_all_entity_types.txt", all_entity_types)

    selected_types, usage = select_relevant_entity_types(
        client=client,
        question=question,
        all_entity_types=all_entity_types,
        model=model,
        llm_logger=llm_logger,
    )
    prompt_usages.append(usage)
    _write_list_txt(question_out_dir, "02_selected_entity_types.txt", selected_types)

    grouped_candidates = entities_by_selected_types(entities, selected_types) if selected_types else {}
    _write_list_txt(
        question_out_dir,
        "03_candidate_entities_filtered_by_type.txt",
        _flatten_grouped_candidates(grouped_candidates),
    )

    if candidate_entity_labels(grouped_candidates):
        selected_entity_names, usage = select_relevant_entities(
            client=client,
            question=question,
            grouped_entities=grouped_candidates,
            model=model,
            llm_logger=llm_logger,
        )
        prompt_usages.append(usage)
    else:
        selected_entity_names = []

    _write_list_txt(question_out_dir, "04_selected_entity_names.txt", selected_entity_names)

    retrieved_model_entities = retrieved_entities_from_names(entities, selected_entity_names)
    _write_list_txt(question_out_dir, "05_retrieved_model_entities.txt", retrieved_model_entities)

    final_answer: FinalAnswer
    elements_included_in_prompt: list[str] = []

    if retrieved_model_entities:
        entity_only_answer, usage, entity_prompt_elements = answer_from_entities_only(
            client=client,
            question=question,
            retrieved_entities=retrieved_model_entities,
            model=model,
            llm_logger=llm_logger,
        )
        prompt_usages.append(usage)
    else:
        entity_only_answer = FinalAnswer(can_answer=False, response="", elements_used=[])
        entity_prompt_elements = []

    _write_text_txt(
        question_out_dir,
        "05a_entity_only_llm_attempt.txt",
        (
            f"can_answer: {entity_only_answer.can_answer}\n\n"
            f"Response:\n{entity_only_answer.response}\n\n"
            f"Elements Used:\n{json.dumps(entity_only_answer.elements_used, ensure_ascii=False, indent=2)}\n\n"
            f"Elements Included In Prompt:\n{json.dumps(entity_prompt_elements, ensure_ascii=False, indent=2)}\n"
        ),
    )

    _write_list_txt(question_out_dir, "06_all_relationship_types.txt", all_relationship_types)

    if entity_only_answer.can_answer:
        final_answer = entity_only_answer
        elements_included_in_prompt = entity_prompt_elements
        _write_list_txt(question_out_dir, "07_selected_relationship_types.txt", [])
        _write_list_txt(question_out_dir, "08_retrieved_relations.txt", [])
    else:
        if not all_relationship_types:
            final_answer = FinalAnswer(
                can_answer=False,
                response="I could not answer the question because the model contains no relationship types.",
                elements_used=[],
            )
            elements_included_in_prompt = []
            _write_list_txt(question_out_dir, "07_selected_relationship_types.txt", [])
            _write_list_txt(question_out_dir, "08_retrieved_relations.txt", [])
        else:
            selected_relationship_types, usage = select_relevant_relationship_types(
                client=client,
                question=question,
                all_relationship_types=all_relationship_types,
                model=model,
                llm_logger=llm_logger,
            )
            prompt_usages.append(usage)
            _write_list_txt(question_out_dir, "07_selected_relationship_types.txt", selected_relationship_types)

            retrieved_relations = retrieve_relations_for_names_and_types(
                relations=relations,
                id_to_entity=id_to_entity,
                id_to_relation=id_to_relation,
                selected_names=selected_entity_names,
                selected_relation_types=selected_relationship_types,
            )
            _write_list_txt(question_out_dir, "08_retrieved_relations.txt", retrieved_relations)

            if retrieved_relations:
                final_answer, usage, relation_prompt_elements = answer_from_relations(
                    client=client,
                    question=question,
                    retrieved_relations=retrieved_relations,
                    model=model,
                    llm_logger=llm_logger,
                )
                prompt_usages.append(usage)
                elements_included_in_prompt = relation_prompt_elements
            else:
                final_answer = FinalAnswer(
                    can_answer=False,
                    response="I could not answer the question because no relevant relations were retrieved.",
                    elements_used=[],
                )
                elements_included_in_prompt = []

            if should_retry_relation_answer(final_answer) and selected_entity_names:
                fallback_relations = retrieve_relations_for_names_and_types(
                    relations=relations,
                    id_to_entity=id_to_entity,
                    id_to_relation=id_to_relation,
                    selected_names=selected_entity_names,
                    selected_relation_types=[],
                )

                if fallback_relations and fallback_relations != retrieved_relations:
                    _write_list_txt(
                        question_out_dir,
                        "08a_retrieved_relations_unfiltered_by_type.txt",
                        fallback_relations,
                    )

                    fallback_answer, usage, fallback_prompt_elements = answer_from_relations(
                        client=client,
                        question=question,
                        retrieved_relations=fallback_relations,
                        model=model,
                        llm_logger=llm_logger,
                    )
                    prompt_usages.append(usage)

                    final_answer = fallback_answer
                    elements_included_in_prompt = fallback_prompt_elements

    _write_text_txt(
        question_out_dir,
        "09_final_llm_response.txt",
        (
            f"can_answer: {final_answer.can_answer}\n\n"
            f"Response:\n{final_answer.response}\n\n"
            f"Elements Used:\n{json.dumps(final_answer.elements_used, ensure_ascii=False, indent=2)}\n\n"
            f"Elements Included In Prompt:\n{json.dumps(elements_included_in_prompt, ensure_ascii=False, indent=2)}"
        ),
    )

    _write_text_txt(
        question_out_dir,
        "10_llm_prompt_token_breakdown.txt",
        render_prompt_usage_report(prompt_usages, question),
    )

    _write_text_txt(
        question_out_dir,
        "10a_llm_prompt_output_file_index.txt",
        render_llm_io_index(llm_logger.saved_files, llm_logger.subdir_name),
    )

    final_output = {
        "Question": question,
        "Response": final_answer.response,
        "Elements Used": final_answer.elements_used,
        "Elements Included In Prompt": elements_included_in_prompt,
    }
    _write_text_txt(
        question_out_dir,
        "11_final_output.txt",
        json.dumps(final_output, ensure_ascii=False, indent=2),
    )

    return QuestionRunResult(
        question=question,
        response=final_answer.response,
        elements_used=final_answer.elements_used,
        elements_included_in_prompt=elements_included_in_prompt,
        prompt_usages=prompt_usages,
    )


# =========================
# JSON question helpers
# =========================

def load_questions_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Questions JSON must be a top-level list of objects.")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx - 1} is not a JSON object.")
        if "Question" not in item or not isinstance(item["Question"], str) or not item["Question"].strip():
            raise ValueError(f"Item at index {idx - 1} is missing a non-empty 'Question' string.")
        normalized.append(item)

    return normalized


def apply_limit(questions: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return questions
    if limit < 0:
        raise ValueError("Question limit must be >= 0.")
    return questions[:limit]


# =========================
# Main
# =========================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer multiple questions over an ArchiMate XML file and write results back into a JSON file."
    )
    parser.add_argument("--xml", default=FILE_PATH, help="Path to the .archimate/.xml file")
    parser.add_argument("--questions-json", default=QUESTIONS_JSON_PATH, help="Path to the input questions JSON file")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Path to the filled output JSON file")
    parser.add_argument("--model", default=MODEL, help="OpenAI model name")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Folder path where outputs will be saved")
    parser.add_argument(
        "--limit",
        type=int,
        default=QUESTION_LIMIT,
        help="Maximum number of questions to process from the input JSON",
    )
    args = parser.parse_args(argv)

    file_path = Path(args.xml)
    questions_json_path = Path(args.questions_json)
    output_json_path = Path(args.output_json)
    out_dir = Path(args.outdir)

    if not file_path.exists():
        print(json.dumps({"error": f"File not found: {file_path}"}, indent=2), file=sys.stderr)
        return 2

    if not questions_json_path.exists():
        print(json.dumps({"error": f"Questions JSON not found: {questions_json_path}"}, indent=2), file=sys.stderr)
        return 2

    try:
        entities, relations, id_to_entity, id_to_relation = parse_archimate_model(file_path)
    except ET.ParseError as exc:
        print(json.dumps({"error": f"XML parse error: {exc}"}, indent=2), file=sys.stderr)
        return 3

    try:
        question_items = load_questions_json(questions_json_path)
        questions_to_process = apply_limit(question_items, args.limit)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 4

    all_entity_types = unique_entity_types(entities)
    all_relationship_types = unique_relation_types(relations)

    try:
        _write_list_txt(out_dir, "00_all_entity_types_global.txt", all_entity_types)
        _write_list_txt(out_dir, "00_all_relationship_types_global.txt", all_relationship_types)
    except OSError as exc:
        print(f"Warning: could not write global outputs to {out_dir}: {exc}", file=sys.stderr)

    client = OpenAI()
    all_question_prompt_usages: list[dict[str, Any]] = []

    for index, item in enumerate(questions_to_process, start=1):
        question = item["Question"]
        question_dir_name = f"question_{index:03d}_{sanitize_filename(question)[:80]}"
        question_out_dir = out_dir / question_dir_name

        try:
            result = process_single_question(
                client=client,
                question=question,
                model=args.model,
                question_out_dir=question_out_dir,
                entities=entities,
                relations=relations,
                id_to_entity=id_to_entity,
                id_to_relation=id_to_relation,
                all_entity_types=all_entity_types,
                all_relationship_types=all_relationship_types,
            )
        except Exception as exc:
            print(f"Warning: failed processing question {index}: {question}\n  {exc}", file=sys.stderr)
            item["Response"] = f"ERROR: {exc}"
            item["Elements Used"] = []
            item["Elements Included In Prompt"] = []
            all_question_prompt_usages.append(
                {
                    "question": question,
                    "prompt_usages": [],
                }
            )
            continue

        item["Response"] = result.response
        item["Elements Used"] = result.elements_used
        item["Elements Included In Prompt"] = result.elements_included_in_prompt

        all_question_prompt_usages.append(
            {
                "question": question,
                "prompt_usages": result.prompt_usages,
            }
        )

    try:
        _write_json(output_json_path, question_items)
    except OSError as exc:
        print(f"Warning: could not write output JSON to {output_json_path}: {exc}", file=sys.stderr)
        return 5

    try:
        _write_text_txt(
            out_dir,
            "00b_all_questions_llm_prompt_token_breakdown.txt",
            render_all_questions_usage_report(all_question_prompt_usages),
        )
        _write_json(
            out_dir / "00c_all_questions_llm_prompt_token_breakdown.json",
            all_question_prompt_usages,
        )
    except OSError as exc:
        print(f"Warning: could not write aggregated token usage report to {out_dir}: {exc}", file=sys.stderr)

    summary = {
        "xml_file": str(file_path),
        "questions_json": str(questions_json_path),
        "output_json": str(output_json_path),
        "processed_questions": len(questions_to_process),
        "total_questions_in_input": len(question_items),
        "results_folder": str(out_dir),
        "model": args.model,
        #"reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
