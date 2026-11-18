"""입력 문장 하나에서 정보와 감성을 함께 분석한다."""

import argparse
import json

from nlp.extract import extract
from nlp.sentiment import analyze


def format_report(text, result, use_rules):
    """분석 결과를 정보 목록과 표현별 계산 과정으로 보여 준다."""
    information, sentiment = result["information"], result["sentiment"]
    lines = ["=== 정보 추출 결과 ===", f"입력: {text}", "", "추출 결과:"]
    for kind, label in (("email", "이메일"), ("phone", "전화"), ("date", "날짜"),
                        ("money", "금액"), ("url", "URL")):
        values = information[kind]
        if kind == "money":
            values = [f"{value['amount']} {value['currency']}" for value in values]
        lines.append(f"- {label}: {', '.join(values) if values else '없음'}")

    lines.extend(["", "=== 감성 분석 과정 ===", f"문장: {text}",
                  f"분석 모드: {'사전 + 부정·강조 규칙' if use_rules else '사전만 (부정·강조 규칙 미적용)'}",
                  f"토큰화: {json.dumps(sentiment['tokens'], ensure_ascii=False)}",
                  "", "감성 표현별 계산:"])
    for index, hit in enumerate(sentiment["hits"], 1):
        lines.extend([
            f"{index}. 표현: {hit['token']} → 사전 키: {hit['term']} ({hit['base']:+g})",
            f"   부정: {hit['negations']}회 / 강조·약화 가중치: ×{hit['weight']:g}",
            f"   계산: ({hit['base']:+g}) × (-1)^{hit['negations']} "
            f"× {hit['weight']:g} = {hit['score']:+g}",
        ])
    if not sentiment["hits"]:
        lines.append("- 사전에 일치하는 감성 표현이 없습니다.")
    lines.extend(["", f"최종 점수: {sentiment['score']:g}",
                  f"최종 판정: {sentiment['label']}"])
    if sentiment["score"] == 0:
        reason = "감성 표현 미탐지" if not sentiment["hits"] else "긍정·부정 점수 상쇄"
        lines.append(f"0점 사유: {reason}")
        if sentiment["label"] == "긍정":
            lines.append("판정 정책: --binary 옵션으로 0점을 긍정으로 처리합니다.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="분석할 텍스트")
    parser.add_argument("--baseline", action="store_true", help="감성 사전 점수만 합산")
    parser.add_argument("--binary", action="store_true", help="0점을 긍정으로 판정해 긍정·부정 중 하나를 출력")
    parser.add_argument("--format", choices=("text", "json"), default="text",
                        help="출력 형식: text는 분석 과정, json은 구조화된 결과 (기본: text)")
    args = parser.parse_args()
    result = {"information": extract(args.text),
              "sentiment": analyze(args.text, use_rules=not args.baseline,
                                   zero_label="긍정" if args.binary else "중립")}
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(args.text, result, use_rules=not args.baseline))


if __name__ == "__main__":
    main()
