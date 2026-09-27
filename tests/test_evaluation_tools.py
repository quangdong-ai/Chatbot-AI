import unittest
from pathlib import Path

import run_benchmark
from evaluation import run_retrieval_eval, validate_golden_set


class GoldenSetValidationTests(unittest.TestCase):
    def test_valid_refusal_case_does_not_need_source(self):
        case = {
            "id": "trap",
            "question": "Ngoai pham vi?",
            "question_type": "out_of_scope",
            "must_answer": False,
            "must_refuse": True,
            "should_have_source": False,
        }
        self.assertEqual(validate_golden_set.validate_case(case, 0), [])

    def test_source_required_case_must_have_target(self):
        case = {
            "id": "missing_source_target",
            "question": "Cau hoi trong tai lieu?",
            "should_have_source": True,
        }
        errors = validate_golden_set.validate_case(case, 0)
        self.assertTrue(any("source target" in error for error in errors))

    def test_duplicate_ids_are_reported(self):
        errors = validate_golden_set.validate_cases([
            {"id": "dup", "question": "A", "should_have_source": False},
            {"id": "dup", "question": "B", "should_have_source": False},
        ])
        self.assertTrue(any("duplicate id" in error for error in errors))

    def test_tailieu_v2_100_fixture_is_valid(self):
        fixture = Path("tests/fixtures/golden_tailieu_v2_100.json")
        cases = validate_golden_set.load_cases(fixture)
        self.assertEqual(len(cases), 100)
        self.assertEqual(validate_golden_set.validate_cases(cases), [])
        self.assertTrue(any(case.get("question_type") == "table" for case in cases))
        self.assertTrue(any(case.get("must_refuse") is True for case in cases))


class RetrievalEvalHelperTests(unittest.TestCase):
    def test_item_match_uses_structured_source_fields(self):
        case = {
            "source_document": "tailieu.pdf",
            "source_page": 6,
            "source_type": "table",
            "expected_article": "Dieu 6",
            "expected_clause": "khoan 2",
        }
        item = {
            "source": "tailieu.pdf",
            "pdf_page": 6,
            "type": "table",
            "article": "Điều 6",
            "clause": "khoản 2",
            "source_line": "Theo tài liệu tailieu.pdf, bảng PDF, trang PDF 6",
            "preview": "Tổng thời gian đào tạo",
        }
        self.assertTrue(run_retrieval_eval.item_matches(case, item))


class AnswerBenchmarkHelperTests(unittest.TestCase):
    def test_expected_source_needles_from_structured_fields(self):
        case = {
            "source_document": "tailieu.pdf",
            "source_page": 6,
            "source_type": "table",
            "expected_article": "Dieu 6",
            "expected_clause": "khoan 2",
        }
        needles = run_benchmark.expected_source_needles(case)
        self.assertIn("tailieu.pdf", needles)
        self.assertIn("trang PDF 6", needles)
        self.assertIn("bảng PDF", needles)
        self.assertIn("Dieu 6", needles)
        self.assertIn("khoan 2", needles)

    def test_summary_reports_refusal_accuracy(self):
        summary = run_benchmark.summarize([
            {
                "ok": True,
                "answer_ok": True,
                "source_ok": True,
                "forbidden_ok": True,
                "top_type_ok": True,
                "refusal_ok": True,
                "must_refuse": True,
                "must_answer": False,
                "should_have_source": False,
                "answer_reference_recall": 1.0,
                "response_time_ms": 10,
            },
            {
                "ok": False,
                "answer_ok": False,
                "source_ok": True,
                "forbidden_ok": True,
                "top_type_ok": True,
                "refusal_ok": True,
                "must_refuse": False,
                "must_answer": True,
                "should_have_source": True,
                "answer_reference_recall": 0.5,
                "response_time_ms": 20,
            },
        ])
        self.assertEqual(summary["refusal_accuracy"], 1.0)
        self.assertEqual(summary["answer_correctness"], 0.0)
        self.assertEqual(summary["citation_correctness"], 1.0)


if __name__ == "__main__":
    unittest.main()
