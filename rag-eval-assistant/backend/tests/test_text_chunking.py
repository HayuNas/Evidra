import unittest

from app.rag.text import chunk_text, extract_text, tokenize
from app.schemas.models import ChunkingConfig


class TestTextChunking(unittest.TestCase):
    def test_extract_text_accepts_markdown_and_text(self):
        content = b"# Policy\n\nReceipts are required."

        self.assertEqual(extract_text("policy.md", content), "# Policy\n\nReceipts are required.")
        self.assertEqual(extract_text("policy.txt", b"plain text"), "plain text")

    def test_extract_text_rejects_unsupported_suffix(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            extract_text("image.png", b"not text")

    def test_chunk_text_preserves_heading_and_stable_ids(self):
        text = "# Finance\nReceipts are required for expenses.\n\n# Security\nRotate keys quarterly."

        chunks = chunk_text(text, source="handbook.md", max_chars=48)

        self.assertEqual(chunks[0].id, "handbook.md:0001")
        self.assertEqual(chunks[0].heading, "Finance")
        self.assertIn("Receipts", chunks[0].text)
        self.assertEqual(chunks[-1].heading, "Security")

    def test_paragraph_chunking_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = chunk_text(text, "doc.md", ChunkingConfig(strategy="paragraph", chunk_size=900))

        self.assertEqual([chunk.text for chunk in chunks], ["First paragraph.", "Second paragraph.", "Third paragraph."])

    def test_fixed_chunking_uses_size_and_overlap(self):
        text = f"{'a' * 300}{'b' * 100}"

        chunks = chunk_text(text, "doc.txt", ChunkingConfig(strategy="fixed", chunk_size=300, overlap=50))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "a" * 300)
        self.assertEqual(chunks[1].text, f"{'a' * 50}{'b' * 100}")

    def test_tokenize_normalizes_words_and_simple_plurals(self):
        self.assertEqual(tokenize("Receipts, receipts! KEY-rotation"), ["receipt", "receipt", "key", "rotation"])


if __name__ == "__main__":
    unittest.main()
