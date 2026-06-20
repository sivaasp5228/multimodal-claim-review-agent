"""
metrics.py — Evaluation metrics for claim verification accuracy.

Computes field-level accuracy, confusion matrices, and detailed
breakdowns for claim_status, issue_type, object_part, severity,
risk_flags, and evidence_standard_met.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    """Load a CSV file as a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def exact_match_accuracy(
    predictions: List[str], ground_truth: List[str]
) -> float:
    """Compute exact match accuracy."""
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def compute_field_accuracy(
    pred_rows: List[Dict[str, str]],
    gt_rows: List[Dict[str, str]],
    field: str,
) -> Tuple[float, List[Tuple[int, str, str]]]:
    """
    Compute accuracy for a single field across all rows.

    Returns:
        (accuracy, list_of_mismatches) where each mismatch is
        (row_index, predicted, expected)
    """
    preds = [r.get(field, "").strip().lower() for r in pred_rows]
    gts = [r.get(field, "").strip().lower() for r in gt_rows]

    mismatches = []
    correct = 0
    for i, (p, g) in enumerate(zip(preds, gts)):
        if p == g:
            correct += 1
        else:
            mismatches.append((i, p, g))

    accuracy = correct / len(gts) if gts else 0.0
    return accuracy, mismatches


def compute_set_field_accuracy(
    pred_rows: List[Dict[str, str]],
    gt_rows: List[Dict[str, str]],
    field: str,
    separator: str = ";",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for a semicolon-separated set field
    (e.g., risk_flags, supporting_image_ids).

    Returns:
        (precision, recall, f1)
    """
    total_precision = 0.0
    total_recall = 0.0
    n = len(gt_rows)

    for pred_row, gt_row in zip(pred_rows, gt_rows):
        pred_set = set(
            v.strip().lower()
            for v in pred_row.get(field, "").split(separator)
            if v.strip()
        )
        gt_set = set(
            v.strip().lower()
            for v in gt_row.get(field, "").split(separator)
            if v.strip()
        )

        if gt_set:
            total_recall += len(pred_set & gt_set) / len(gt_set)
        else:
            total_recall += 1.0 if not pred_set else 0.0

        if pred_set:
            total_precision += len(pred_set & gt_set) / len(pred_set)
        else:
            total_precision += 1.0 if not gt_set else 0.0

    precision = total_precision / n if n else 0.0
    recall = total_recall / n if n else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute a confusion matrix.

    Returns:
        Nested dict: matrix[actual][predicted] = count
    """
    if labels is None:
        labels = sorted(set(predictions) | set(ground_truth))

    matrix: Dict[str, Dict[str, int]] = {
        actual: {pred: 0 for pred in labels} for actual in labels
    }

    for p, g in zip(predictions, ground_truth):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1

    return matrix


def format_confusion_matrix(
    matrix: Dict[str, Dict[str, int]], labels: List[str]
) -> str:
    """Format confusion matrix as a readable string."""
    # Header
    col_width = max(len(l) for l in labels) + 2
    header = "Actual \\ Predicted".ljust(col_width)
    header += "".join(l.ljust(col_width) for l in labels)
    lines = [header, "-" * len(header)]

    for actual in labels:
        row = actual.ljust(col_width)
        for pred in labels:
            count = matrix.get(actual, {}).get(pred, 0)
            row += str(count).ljust(col_width)
        lines.append(row)

    return "\n".join(lines)


def compute_all_metrics(
    pred_rows: List[Dict[str, str]],
    gt_rows: List[Dict[str, str]],
) -> Dict:
    """
    Compute all evaluation metrics.

    Returns a dict with accuracy for each field, confusion matrices,
    and detailed mismatch info.
    """
    results = {}

    # ── Exact-match fields ──────────────────────────────────────────────
    exact_fields = [
        "claim_status",
        "issue_type",
        "object_part",
        "severity",
        "evidence_standard_met",
        "valid_image",
    ]
    for field in exact_fields:
        acc, mismatches = compute_field_accuracy(pred_rows, gt_rows, field)
        results[field] = {
            "accuracy": acc,
            "correct": len(gt_rows) - len(mismatches),
            "total": len(gt_rows),
            "mismatches": mismatches,
        }

    # ── Set-match fields ────────────────────────────────────────────────
    set_fields = ["risk_flags", "supporting_image_ids"]
    for field in set_fields:
        p, r, f1 = compute_set_field_accuracy(pred_rows, gt_rows, field)
        results[field] = {
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    # ── Confusion matrix for claim_status ───────────────────────────────
    status_labels = ["supported", "contradicted", "not_enough_information"]
    pred_statuses = [r.get("claim_status", "").strip().lower() for r in pred_rows]
    gt_statuses = [r.get("claim_status", "").strip().lower() for r in gt_rows]
    results["claim_status_confusion"] = confusion_matrix(
        pred_statuses, gt_statuses, status_labels
    )

    return results


def format_report(results: Dict, strategy_name: str = "Default") -> str:
    """Format evaluation results as a markdown report section."""
    lines = [f"### Strategy: {strategy_name}\n"]

    # Field accuracies
    lines.append("| Field | Accuracy | Correct / Total |")
    lines.append("|---|---|---|")
    for field in [
        "claim_status", "issue_type", "object_part",
        "severity", "evidence_standard_met", "valid_image",
    ]:
        data = results.get(field, {})
        acc = data.get("accuracy", 0)
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        lines.append(f"| {field} | {acc:.1%} | {correct}/{total} |")

    lines.append("")

    # Set field metrics
    lines.append("| Set Field | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for field in ["risk_flags", "supporting_image_ids"]:
        data = results.get(field, {})
        p = data.get("precision", 0)
        r = data.get("recall", 0)
        f1 = data.get("f1", 0)
        lines.append(f"| {field} | {p:.1%} | {r:.1%} | {f1:.1%} |")

    lines.append("")

    # Confusion matrix
    lines.append("#### claim_status Confusion Matrix\n")
    lines.append("```")
    cm = results.get("claim_status_confusion", {})
    labels = ["supported", "contradicted", "not_enough_information"]
    lines.append(format_confusion_matrix(cm, labels))
    lines.append("```\n")

    # Mismatches
    for field in [
        "claim_status", "issue_type", "object_part", "severity",
    ]:
        mismatches = results.get(field, {}).get("mismatches", [])
        if mismatches:
            lines.append(f"#### {field} Mismatches\n")
            lines.append("| Row | Predicted | Expected |")
            lines.append("|---|---|---|")
            for idx, pred, exp in mismatches:
                lines.append(f"| {idx + 1} | {pred} | {exp} |")
            lines.append("")

    return "\n".join(lines)
