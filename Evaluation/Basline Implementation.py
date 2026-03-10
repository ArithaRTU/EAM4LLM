from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


# Defaults (can be overridden by CLI args)
FILE_PATH = ""
QUESTIONS_JSON_PATH = ""
DEFAULT_OUTDIR = ""
DEFAULT_OUTPUT_JSON = ""
LLM_IO_SUBDIR = "llm_io"
QUESTION_LIMIT = 0

# LLM related Defaults (can be overridden by CLI args)
MODEL = "gpt-4.1"
#REASONING_EFFORT = "none" relavant for GPT-5.2, but not for GPT-4.1
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


def format_entity_label(name: str, type_name: str) -> str:
    return f"{name} (type: {type_name})"


def all_entity_labels(entities: list[EntityRecord]) -> list[str]:
    labels = {
        format_entity_label(entity.name, entity.type_name)
        for entity in entities
        if entity.name and entity.type_name
    }
    return sorted(labels, key=lambda x: x.casefold())


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


def build_all_relation_strings(
    relations: list[RelationRecord],
    id_to_entity: dict[str, EntityRecord],
    id_to_relation: dict[str, RelationRecord],
) -> list[str]:
    label_memo: dict[str, str] = {}
    seen: set[str] = set()
    results: list[str] = []

    for rel in relations:
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
        relation_name = (rel.relation_name or "").strip()

        if relation_name:
            value = f"{source_label} {rel.relation_type} [{relation_name}] {target_label}"
        else:
            value = f"{source_label} {rel.relation_type} {target_label}"

        if value not in seen:
            seen.add(value)
            results.append(value)

    results.sort(key=lambda x: x.casefold())
    return results



# Token usage helpers


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



# OpenAI helpers


def call_structured_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    prompt_name: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[FinalAnswer, dict[str, Any]]:
    response = client.responses.parse(
        model=model,
#        reasoning={"effort": REASONING_EFFORT},
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=FinalAnswer,
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


def build_single_question_prompt(
    question: str,
    xml_text: str,
    allowed_entity_strings: list[str],
    allowed_relation_strings: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "You are a business analyst answering questions using only the provided ArchiMate model data.\n"
        "You will receive canonical non-XML strings derived from an XML exported from the Archi Tool.\n"
        "Your job is to answer the question strictly from that model data.\n"
        "Do not use external knowledge.\n"
        "Do not invent entities, relationships, names, or facts.\n"
        "The response must be a single paragraph.\n"
        "When referring to entities in the response, include entity names and types in the exact form 'Name (type: Type)' whenever relevant.\n"
        "If the question cannot be answered from the provided model data, set can_answer=false.\n"
        "elements_used must contain only exact strings from the provided allowed lists.\n"
        "elements_used may include entity strings and/or relation strings.\n"
        "Do not return XML in elements_used.\n"
        "Preserve exact string formatting for every value in elements_used."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        "Allowed entity strings for elements_used:\n"
        f"{json.dumps(allowed_entity_strings, ensure_ascii=False, indent=2)}\n\n"
        "Allowed relation strings for elements_used:\n"
        f"{json.dumps(allowed_relation_strings, ensure_ascii=False, indent=2)}\n\n"
    )

    return system_prompt, user_prompt


def answer_question_with_single_prompt(
    client: OpenAI,
    question: str,
    xml_text: str,
    allowed_entity_strings: list[str],
    allowed_relation_strings: list[str],
    model: str,
    llm_logger: LLMArtifactLogger | None = None,
) -> tuple[FinalAnswer, dict[str, Any], list[str]]:
    system_prompt, user_prompt = build_single_question_prompt(
        question=question,
        xml_text=xml_text,
        allowed_entity_strings=allowed_entity_strings,
        allowed_relation_strings=allowed_relation_strings,
    )

    parsed, usage = call_structured_llm(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        prompt_name="single_prompt_answer",
        llm_logger=llm_logger,
    )

    entity_set = set(allowed_entity_strings)
    allowed_all = allowed_entity_strings + [r for r in allowed_relation_strings if r not in entity_set]
    sanitized_elements_used = sanitize_exact_strings(parsed.elements_used, allowed_all)

    final_answer = FinalAnswer(
        can_answer=parsed.can_answer,
        response=parsed.response.strip(),
        elements_used=sanitized_elements_used if parsed.can_answer else [],
    )

    return final_answer, usage, allowed_all

# Question processing

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
    xml_text: str,
    all_entity_strings: list[str],
    all_relation_strings: list[str],
) -> QuestionRunResult:
    llm_logger = LLMArtifactLogger(out_dir=question_out_dir)
    prompt_usages: list[dict[str, Any]] = []

    _write_list_txt(question_out_dir, "01_all_entity_strings.txt", all_entity_strings)
    _write_list_txt(question_out_dir, "02_all_relation_strings.txt", all_relation_strings)
    _write_text_txt(question_out_dir, "03_xml_included_in_prompt.archimate.xml", xml_text)

    final_answer, usage, elements_included_in_prompt = answer_question_with_single_prompt(
        client=client,
        question=question,
        xml_text=xml_text,
        allowed_entity_strings=all_entity_strings,
        allowed_relation_strings=all_relation_strings,
        model=model,
        llm_logger=llm_logger,
    )
    prompt_usages.append(usage)

    _write_text_txt(
        question_out_dir,
        "04_final_llm_response.txt",
        (
            f"can_answer: {final_answer.can_answer}\n\n"
            f"Response:\n{final_answer.response}\n\n"
            f"Elements Used:\n{json.dumps(final_answer.elements_used, ensure_ascii=False, indent=2)}\n\n"
            f"Elements Included In Prompt:\n{json.dumps(elements_included_in_prompt, ensure_ascii=False, indent=2)}"
        ),
    )

    _write_text_txt(
        question_out_dir,
        "05_llm_prompt_token_breakdown.txt",
        render_prompt_usage_report(prompt_usages, question),
    )

    _write_text_txt(
        question_out_dir,
        "05a_llm_prompt_output_file_index.txt",
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
        "06_final_output.txt",
        json.dumps(final_output, ensure_ascii=False, indent=2),
    )

    return QuestionRunResult(
        question=question,
        response=final_answer.response,
        elements_used=final_answer.elements_used,
        elements_included_in_prompt=elements_included_in_prompt,
        prompt_usages=prompt_usages,
    )



# JSON question helpers


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



# Main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer multiple questions over an ArchiMate XML file using one LLM prompt per question."
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
        xml_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(json.dumps({"error": f"Could not read XML file: {exc}"}, indent=2), file=sys.stderr)
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
    all_entity_strings = all_entity_labels(entities)
    all_relation_string_list = build_all_relation_strings(relations, id_to_entity, id_to_relation)

    try:
        _write_list_txt(out_dir, "00_all_entity_types_global.txt", all_entity_types)
        _write_list_txt(out_dir, "00_all_relationship_types_global.txt", all_relationship_types)
        _write_list_txt(out_dir, "00a_all_entity_strings_global.txt", all_entity_strings)
        _write_list_txt(out_dir, "00b_all_relation_strings_global.txt", all_relation_string_list)
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
                xml_text=xml_text,
                all_entity_strings=all_entity_strings,
                all_relation_strings=all_relation_string_list,
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
            "00c_all_questions_llm_prompt_token_breakdown.txt",
            render_all_questions_usage_report(all_question_prompt_usages),
        )
        _write_json(
            out_dir / "00d_all_questions_llm_prompt_token_breakdown.json",
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
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())