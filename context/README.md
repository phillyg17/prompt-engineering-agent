# Context Folder

Drop any PDF (`.pdf`) or Microsoft Word (`.docx`) documents here.

The **Document Loader Agent** will parse every supported file in this folder before the writing phase begins. Extracted text is merged with any written context you provide via the CLI and fed to the Prompt Writer.

## Supported formats

- `.pdf` — text extracted page-by-page
- `.docx` — paragraphs extracted in order

## Usage ideas

- Upload a **style guide** so the generated prompt matches your team's tone.
- Upload **example outputs** (as docs) so the prompt can request similar structure.
- Upload **requirements documents** so the prompt captures all necessary constraints.

> **Note:** Sub-folders are not scanned (only the top-level `context/` directory). Unsupported file types are silently skipped.
