# AI-readable package: pre-revision thesis and blind-review comments

Created at: 2026-04-30 18:17:12

This folder is intended for future AI assistants. It avoids repeated PDF extraction, encoding detection, and page splitting.

## Source files

- Pre-revision thesis PDF: `D:\xidian_Master\研究生论文\毕业论文\xduts-main\明审版本\xdupgthesis_template_lc.pdf`
- Official returned review PDF: `D:\xidian_Master\研究生论文\毕业论文\xduts-main\刘畅_23011211000_面向智慧交通的车型分类及停车占用检测方法研究_1.pdf`
- Review/comment TXT: `D:\xidian_Master\研究生论文\毕业论文\xduts-main\修改意见.txt`

## Outputs

- `00_manifest.json`: machine-readable index of all generated files.
- `01_thesis_before_full_layout.txt`: pre-revision thesis, layout-preserving text.
- `01_thesis_before_full_plain.txt`: pre-revision thesis, plain text.
- `01_thesis_before_pages.jsonl`: pre-revision thesis split by PDF page.
- `02_review_return_pdf_full_layout.txt`: official review PDF, layout-preserving text.
- `02_review_return_pdf_full_plain.txt`: official review PDF, plain text.
- `02_review_return_pdf_pages.jsonl`: official review PDF split by PDF page.
- `03_review_comments_txt_utf8.md`: UTF-8 Markdown copy of the review/comment TXT.
- `03_review_comments_txt_sections.jsonl`: review/comment TXT split by Markdown headings.
- `04_AI_read_this_first.md`: short reading guide for future AI assistants.

Use the JSONL files for chunked retrieval and the layout TXT files for manual checking of tables and page structure.
