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
                  f"1. 토큰화: [{', '.join(sentiment['tokens'])}]"])
    hits = sentiment["hits"]
    lines.append("2. 감성 단어 탐지:")
    for hit in hits:
        lines.append(f"   - {hit['token']} → {hit['term']} ({hit['base']:+g})")
    if not hits:
        lines.append("   - 없음")
    lines.append("3. 부정어 탐지:")
    for hit in hits:
        terms = ", ".join(f"'{term}'" for term in hit["negation_terms"])
        effect = "감성 반전" if hit["negations"] % 2 else "원래 방향 유지"
        lines.append(f"   - {hit['token']}: {terms or '적용 없음'}"
                     + (f" → {effect} ({hit['negations']}회)" if terms else ""))
    if not hits:
        lines.append("   - 적용 없음")
    lines.append("4. 강조어 탐지:")
    for hit in hits:
        term = f"'{hit['intensifier']}' → " if hit["intensifier"] else "적용 없음 → "
        lines.append(f"   - {hit['token']}: {term}강도 ×{hit['weight']:g}")
    if not hits:
        lines.append("   - 적용 없음")
    for hit in hits:
        sign = "(-1)" if hit["negations"] % 2 else "(+1)"
        lines.append(f"계산: {hit['token']} · ({hit['base']:+g}) × {sign} "
                     f"× {hit['weight']:g} = {hit['score']:+g}")
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
                                   zero_label="긍정" if args.binary else "중립",
                                   include_rule_terms=args.format == "text")}
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(args.text, result, use_rules=not args.baseline))


if __name__ == "__main__":
    main()
