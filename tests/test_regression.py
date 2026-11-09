"""정보 추출·감성 분석·평가의 핵심 동작을 검증한다."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from evaluate import evaluate_extraction, evaluate_sentiment, main as evaluate_main, read_data
from nlp.extract import extract
from nlp.sentiment import analyze

ROOT = Path(__file__).resolve().parents[1]


class ExtractionTests(unittest.TestCase):
    def test_supported_variants_and_normalization(self):
        cases = {
            "email": [
                ("user@example.com", "user@example.com"),
                ("a.b@SUB.Example.com", "a.b@sub.example.com"),
                ("a+b@example.org", "a+b@example.org"),
            ],
            "phone": [
                ("(02) 123-4567", "02-123-4567"),
                ("031 1234 5678", "031-1234-5678"),
                ("01012345678", "010-1234-5678"),
                ("+82 10-1234-5678", "010-1234-5678"),
                ("+82-2-123-4567", "02-123-4567"),
                ("공일공-일이삼사-오육칠팔입니다", "010-1234-5678"),
                ("영이-일이삼-사오육칠", "02-123-4567"),
            ],
            "date": [
                ("2024년 2월 29일", "2024-02-29"),
                ("2024/2/29", "2024-02-29"),
                ("2024-02-29", "2024-02-29"),
            ],
            "money": [
                ("10,000원", {"amount": 10000, "currency": "KRW"}),
                ("1억 2천만원", {"amount": 120000000, "currency": "KRW"}),
                ("$100", {"amount": 100, "currency": "USD"}),
            ],
            "url": [
                ("http://EXAMPLE.com/a", "http://example.com/a"),
                ("https://example.com/a?q=1", "https://example.com/a?q=1"),
                ("www.example.com.", "https://www.example.com"),
            ],
        }
        for kind, examples in cases.items():
            for raw, expected in examples:
                with self.subTest(kind=kind, raw=raw):
                    self.assertEqual(extract(raw)[kind], [expected])

    def test_invalid_and_overlapping_matches(self):
        result = extract(
            "2023-02-29 2024/12-31 10,00원 https://example.com/2024-01-01 "
            "공일공-일이삼사-오육칠"
        )
        self.assertEqual(result["date"], [])
        self.assertEqual(result["money"], [])
        self.assertEqual(result["phone"], [])
        self.assertEqual(result["url"], ["https://example.com/2024-01-01"])

    def test_extraction_evaluation_fixture(self):
        counts, failures = evaluate_extraction(read_data("extraction.jsonl"))
        self.assertEqual(dict(counts["email"]), {"tp": 10, "fp": 0, "fn": 2})
        self.assertEqual(dict(counts["phone"]), {"tp": 10, "fp": 0, "fn": 1})
        self.assertEqual(dict(counts["url"]), {"tp": 9, "fp": 1, "fn": 2})
        self.assertEqual(len(failures), 9)


class SentimentTests(unittest.TestCase):
    def test_negation_intensification_and_scope(self):
        cases = [
            ("안 좋다", -2, 1, 1),
            ("좋지 않다", -2, 1, 1),
            ("좋지 않지 않다", 2, 2, 1),
            ("좋지는 않지는 않아요", 2, 2, 1),
            ("정말 안 좋다", -3, 1, 1.5),
            ("완전 정말 좋다", 3, 0, 1.5),
            ("조금 좋다", 1, 0, 0.5),
            ("좋지, 않다", 2, 0, 1),
        ]
        for text, score, negations, weight in cases:
            with self.subTest(text=text):
                result = analyze(text)
                self.assertEqual(result["score"], score)
                self.assertEqual(len(result["hits"]), 1)
                self.assertEqual(result["hits"][0]["negations"], negations)
                self.assertEqual(result["hits"][0]["weight"], weight)

    def test_phrase_and_label_policy(self):
        result = analyze("완전 시간 낭비")
        self.assertEqual(result["score"], -4)
        self.assertEqual([hit["term"] for hit in result["hits"]], ["시간 낭비"])
        self.assertEqual(analyze("좋다. 나쁘다.")["label"], "중립")
        self.assertEqual(analyze("좋다. 나쁘다.", zero_label="긍정")["label"], "긍정")
        self.assertEqual(analyze("정말 좋지 않다", use_rules=False)["score"], 2)

    def test_sentiment_evaluation_metrics(self):
        rows = [{"text": "좋다", "label": "긍정"},
                {"text": "좋지 않다", "label": "부정"}]
        accuracy, f1, failures = evaluate_sentiment(rows, use_rules=True)
        self.assertEqual((accuracy, f1, failures), (1.0, {"긍정": 1.0, "부정": 1.0}, []))
        self.assertEqual(evaluate_sentiment(rows, use_rules=False)[0], 0.5)


class CLITests(unittest.TestCase):
    def test_combined_json_output(self):
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--text", "정말 좋지 않다. (02) 123-4567"],
            cwd=ROOT, check=True, capture_output=True, encoding="utf-8",
        )
        result = json.loads(process.stdout)
        self.assertEqual(result["information"]["phone"], ["02-123-4567"])
        self.assertEqual(result["sentiment"]["label"], "부정")
        self.assertEqual(result["sentiment"]["score"], -3)

    def test_full_evaluation_reports_both_settings(self):
        output = StringIO()
        with patch.object(sys, "argv", ["evaluate.py", "--errors", "0"]), redirect_stdout(output):
            evaluate_main()
        report = output.getvalue()
        self.assertIn("정보 추출 평가 (60문장)", report)
        self.assertIn("NSMC 감성 분석 평가 (49997문장", report)
        self.assertIn("| 사전만 |", report)
        self.assertIn("| 사전 + 규칙 |", report)
