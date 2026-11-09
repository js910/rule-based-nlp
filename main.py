"""입력 문장 하나에서 정보와 감성을 함께 분석한다."""

import argparse
import json

from nlp.extract import extract
from nlp.sentiment import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="분석할 텍스트")
    parser.add_argument("--baseline", action="store_true", help="감성 사전 점수만 합산")
    parser.add_argument("--binary", action="store_true", help="0점을 긍정으로 판정해 긍정·부정 중 하나를 출력")
    args = parser.parse_args()
    result = {"information": extract(args.text),
              "sentiment": analyze(args.text, use_rules=not args.baseline,
                                   zero_label="긍정" if args.binary else "중립")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
