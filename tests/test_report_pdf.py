"""Batch PDF conversion preserves report sources and link destinations."""

import tempfile
import unittest
from pathlib import Path

from markdown_it import MarkdownIt

from utils.report_pdf import _inline, convert_markdown_reports


class ReportPdfTests(unittest.TestCase):
    def test_local_report_links_change_to_pdf_and_external_links_are_preserved(self):
        tokens = MarkdownIt().parseInline(
            '[Parish](001_parish.md) and [Evidence](https://example.org/?a=1&b=2)'
        )[0].children
        markup = _inline(tokens)
        self.assertIn('href="001_parish.pdf"', markup)
        self.assertIn('href="https://example.org/?a=1&amp;b=2"', markup)

    def test_batch_includes_index_and_preserves_markdown_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                'index.md': '# Reports\n\n[Parish](001_parish.md)\n',
                '001_parish.md': '# Saint José\n\nNo ministry labels recorded.\n\n'
                '| Ministry | Evidence |\n| --- | --- |\n'
                '| Food & Care | [Source](https://example.org) |\n',
            }
            for name, content in sources.items():
                (root / name).write_text(content, encoding='utf-8')
            (root / 'pdf').mkdir()
            (root / 'pdf' / 'unrelated.txt').write_text('retain', encoding='utf-8')
            for _ in range(2):
                manifest = convert_markdown_reports(root)
                self.assertEqual(len(manifest), 2)
                for row in manifest:
                    self.assertTrue(Path(row['pdf_path']).read_bytes().startswith(b'%PDF-'))
            for name, content in sources.items():
                self.assertEqual((root / name).read_text(encoding='utf-8'), content)
            self.assertTrue((root / 'pdf' / 'unrelated.txt').exists())

    def test_empty_input_raises_instead_of_reporting_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'No Markdown reports'):
                convert_markdown_reports(directory)


if __name__ == '__main__':
    unittest.main()
