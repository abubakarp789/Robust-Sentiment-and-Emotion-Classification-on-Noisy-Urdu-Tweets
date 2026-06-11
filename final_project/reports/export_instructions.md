# Markdown Report Export Instructions

This guide outlines three standard methods (Pandoc, VS Code Extensions, and Manual Word Importing) for converting the markdown project reports (such as `final_report.md`, `dataset_card.md`, and `ethics_and_limitations.md`) into submission-ready formats (PDF or Microsoft Word DOCX).

---

## Method 1: Using Pandoc (Command Line)

[Pandoc](https://pandoc.org/) is a powerful open-source document converter. It is the recommended command-line method for compiling markdown documents.

### Prerequisites
1. Install Pandoc:
   - On Windows (via winget): `winget install mdq.pandoc` or download the installer from GitHub.
   - On macOS: `brew install pandoc`
2. Install a PDF Engine (required for PDF exports only):
   - Install **wkhtmltopdf** (simplest, HTML-based engine) or **MikTeX / MacTeX** (for LaTeX-based premium PDF rendering).

### Conversion Commands

To compile files, open a terminal in the `reports/` folder:

1. **Convert Markdown to Microsoft Word (DOCX)**:
   ```bash
   pandoc final_report.md -o final_report.docx
   ```

2. **Convert Markdown to PDF (using wkhtmltopdf)**:
   ```bash
   pandoc final_report.md -o final_report.pdf --pdf-engine=wkhtmltopdf
   ```

3. **Convert Markdown to PDF (using LaTeX/MikTeX)**:
   ```bash
   pandoc final_report.md -o final_report.pdf --pdf-engine=xelatex
   ```

---

## Method 2: Using VS Code Extensions (Graphical UI)

If you are using Visual Studio Code, you can convert documents directly using extensions without running terminal commands.

### Option A: Markdown PDF (Best for Quick PDF Export)
1. Search for and install the **Markdown PDF** extension (by yzane) in VS Code.
2. Open `final_report.md`.
3. Right-click anywhere in the editor area.
4. Select **Markdown PDF: Export (pdf)**.
5. The extension will generate a formatted PDF file in the same directory.

### Option B: Pandoc PDF/Word (Best for DOCX and LaTeX PDFs)
1. Install the **vscode-pandoc** extension (by Doug Finke) in VS Code.
2. Ensure `pandoc` is installed on your system.
3. Open `final_report.md`.
4. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) to open the Command Palette.
5. Type `Pandoc Render` and hit Enter.
6. Select your target output format: `docx`, `pdf`, or `html`.

---

## Method 3: Manual Import into Microsoft Word

If you need to apply custom corporate or academic word templates, you can import markdown files directly into MS Word.

### Steps
1. Open the `.md` report file in any text editor (VS Code, Notepad, etc.).
2. Select all text (`Ctrl + A`) and copy it (`Ctrl + C`).
3. Open Microsoft Word.
4. Paste the text into a blank document.
5. Highlight markdown elements (e.g., `# Header 1` and `## Header 2`) and assign them Word styles (**Heading 1**, **Heading 2**).
6. Convert markdown bullet lists and tables (MS Word usually parses markdown tables automatically upon pasting).
7. Save the document as `.docx` or export it to PDF using **File > Save As > PDF**.
