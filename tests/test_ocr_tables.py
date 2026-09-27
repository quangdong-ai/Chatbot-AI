import unittest
import unicodedata

import app


def fold(text: str) -> str:
    text = str(text or "").casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d")


class OcrTableTests(unittest.TestCase):
    def scan_ocr_tables(self):
        return [
            doc for doc in app.all_indexed_docs
            if doc.metadata.get("source") == "1790131198_scan.pdf"
            and doc.metadata.get("type") == "ocr_table"
        ]

    def test_scan_pdf_has_expected_ocr_tables(self):
        docs = self.scan_ocr_tables()
        pages = sorted(doc.metadata.get("pdf_page") for doc in docs)
        self.assertEqual(pages, [1, 2])

    def test_scan_ocr_tables_have_confidence_metadata(self):
        for doc in self.scan_ocr_tables():
            with self.subTest(page=doc.metadata.get("pdf_page")):
                self.assertIn("ocr_confidence", doc.metadata)
                self.assertIn("ocr_min_confidence", doc.metadata)
                self.assertGreaterEqual(float(doc.metadata["ocr_confidence"]), 70.0)
                self.assertFalse(doc.metadata.get("ocr_needs_review"))
                self.assertEqual(doc.metadata.get("ocr_validation_notes", ""), "")

    def test_scan_hardware_table_contains_verified_cells(self):
        page_one = next(doc for doc in self.scan_ocr_tables() if doc.metadata.get("pdf_page") == 1)
        content = page_one.page_content
        self.assertIn("NVIDIA GeForce RTX 3050 Ti Laptop GPU", content)
        self.assertIn("4 GB VRAM", content)
        self.assertIn("16 GB", content)
        self.assertIn("3200 MT/s", content)
        self.assertIn("512 GB SSD NVMe", content)

        self.assertNotIn("RTX 3030", content)
        self.assertNotIn("312 GB", content)
        self.assertNotIn("55D", content)

    def test_scan_software_table_contains_verified_cells(self):
        page_two = next(doc for doc in self.scan_ocr_tables() if doc.metadata.get("pdf_page") == 2)
        content = page_two.page_content
        self.assertIn("Windows 11 & WSL2 docker", content)
        self.assertIn("25H2 & v24.0.5", content)
        self.assertIn("Python", content)
        self.assertIn("3.10.9", content)

    def test_scan_table_questions_answer_from_ocr_table(self):
        cases = [
            (
                "Trong scan.pdf, bang 4.1 GPU su dung loai nao va co bao nhieu VRAM?",
                ["RTX 3050 Ti", "4 GB VRAM", "scan.pdf", "bảng OCR"],
            ),
            (
                "Trong scan.pdf, bang 4.1 RAM co dung luong va toc do bao nhieu?",
                ["16 GB", "3200 MT/s", "scan.pdf", "bảng OCR"],
            ),
            (
                "Trong scan.pdf, RAM tối thiểu là bao nhiêu?",
                ["16 GB", "3200 MT/s", "scan.pdf", "bảng OCR"],
            ),
            (
                "Trong scan.pdf, bang 4.1 luu tru dung loai gi va dung luong bao nhieu?",
                ["512 GB SSD NVMe", "scan.pdf", "bảng OCR"],
            ),
            (
                "Trong scan.pdf, bang 4.2 moi truong co so va ao hoa dung cong nghe gi, phien ban bao nhieu?",
                ["Windows 11", "WSL2 docker", "25H2", "v24.0.5", "scan.pdf"],
            ),
            (
                "Trong scan.pdf, bang 4.2 nen tang thuc thi loi dung Python phien ban nao?",
                ["Python", "3.10.9", "scan.pdf"],
            ),
        ]
        for question, expected_items in cases:
            with self.subTest(question=question):
                answer = app._chat_for_eval(question, 1)
                folded_answer = fold(answer)
                for expected in expected_items:
                    self.assertIn(fold(expected), folded_answer)

    def test_scan_retrieval_prioritizes_ocr_table(self):
        result = app._debug_retrieve(
            "Trong scan.pdf, bang 4.1 GPU su dung loai nao va co bao nhieu VRAM?",
            top_k=5,
            bot_id=1,
        )
        self.assertTrue(result["items"])
        first = result["items"][0]
        self.assertEqual(first["source"], "1790131198_scan.pdf")
        self.assertEqual(first["type"], "ocr_table")
        self.assertIn("scan.pdf", first["source_line"])


if __name__ == "__main__":
    unittest.main()
