"""고정된 레이블 데이터로 평가하고 문장별 오류를 출력한다."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from nlp.extract import PATTERNS, extract
from nlp.sentiment import analyze

DATA = Path(__file__).parent / "data"
LABELS = ("긍정", "부정")


def read_data(filename):
    with (DATA / filename).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_nsmc():
    path = DATA / "ratings_test.txt"
    if not path.exists():
        raise SystemExit("NSMC 테스트 데이터가 필요합니다. 먼저 python setup_data.py를 실행하세요.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        # 따옴표도 리뷰 원문이므로 CSV 인용 문법을 비활성화한다.
        rows = csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [{"id": row["id"], "text": row["document"],
                 "label": "긍정" if row["label"] == "1" else "부정"}
                for row in rows if row["document"] and row["document"].strip()]


def as_counter(values):
    return Counter(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate_extraction(rows):
    counts = {kind: Counter() for kind in PATTERNS}
    failures = []
    for row in rows:
        result = extract(row["text"])
        for kind, expected in row["expected"].items():
            predicted = result[kind]
            gold, actual = as_counter(expected), as_counter(predicted)
            counts[kind]["tp"] += sum((gold & actual).values())
            counts[kind]["fp"] += sum((actual - gold).values())
            counts[kind]["fn"] += sum((gold - actual).values())
            if gold != actual:
                failures.append((row, kind, expected, predicted))
    return counts, failures


def evaluate_sentiment(rows, use_rules, zero_label="중립"):
    confusion = Counter()
    failures = []
    for row in rows:
        result = analyze(row["text"], use_rules=use_rules, zero_label=zero_label)
        confusion[row["label"], result["label"]] += 1
        if result["label"] != row["label"]:
            failures.append((row, result))

    accuracy = ratio(sum(confusion[label, label] for label in LABELS), len(rows))
    f1 = {}
    for label in LABELS:
        tp = confusion[label, label]
        fp = sum(value for (gold, pred), value in confusion.items()
                 if pred == label and gold != label)
        fn = sum(value for (gold, pred), value in confusion.items()
                 if gold == label and pred != label)
        f1[label] = ratio(2 * tp, 2 * tp + fp + fn)
    return accuracy, f1, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=int, default=20, help="출력할 감성 오분류 건수")
    parser.add_argument("--binary", action="store_true", help="0점을 긍정으로 판정해 이진 평가")
    args = parser.parse_args()
    if args.errors < 0:
        parser.error("--errors는 0 이상이어야 합니다.")
    rows = read_data("extraction.jsonl")
    counts, failures = evaluate_extraction(rows)
    print(f"정보 추출 평가 ({len(rows)}문장)")
    print("| 유형 | TP | FP | FN | Precision | Recall |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for kind, count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        print(f"| {kind} | {tp} | {fp} | {fn} | "
              f"{ratio(tp, tp + fp):.1%} | {ratio(tp, tp + fn):.1%} |")
    print(f"\n추출 오류 {len(failures)}건")
    for row, kind, expected, predicted in failures:
        print(f"[{row['id']} {kind}] {row['text']} · 정답 {expected} · 출력 {predicted}")
        if "note" in row:
            print(f"  {row['note']}")

    rows = read_nsmc()
    labels = Counter(row["label"] for row in rows)
    print(f"\nNSMC 감성 분석 평가 ({len(rows)}문장, 긍정 {labels['긍정']}, 부정 {labels['부정']})")
    print("| 설정 | 정답 수 | Accuracy | 긍정 F1 | 부정 F1 | Macro F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mode, use_rules in (("사전만", False), ("사전 + 규칙", True)):
        accuracy, f1, failures = evaluate_sentiment(
            rows, use_rules, zero_label="긍정" if args.binary else "중립")
        macro = sum(f1.values()) / len(LABELS)
        print(f"| {mode} | {len(rows) - len(failures)} | {accuracy:.1%} | "
              f"{f1['긍정']:.3f} | {f1['부정']:.3f} | {macro:.3f} |")
    print(f"\n규칙 적용 후 오분류 {len(failures)}건")
    for row, result in failures[:args.errors]:
        terms = ", ".join(f"{hit['term']}={hit['score']:+g}" for hit in result["hits"])
        print(f"[{row['id']}] {row['text']} · 정답 {row['label']} · "
              f"출력 {result['label']} ({terms or '매칭 없음'})")


if __name__ == "__main__":
    main()
