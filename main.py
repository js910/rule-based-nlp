"""입력 문장 하나에서 정보와 감성을 함께 분석한다."""

import argparse
import json

from extract import extract
from sentiment import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="분석할 텍스트")
    parser.add_argument("--baseline", action="store_true", help="감성 사전 점수만 합산")
    args = parser.parse_args()
    result = {"information": extract(args.text),
              "sentiment": analyze(args.text, use_rules=not args.baseline)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
