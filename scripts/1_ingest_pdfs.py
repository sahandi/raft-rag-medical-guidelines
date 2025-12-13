from pathlib import Path
from docling.document_converter import DocumentConverter

PROJECT_ROOT = Path("/Volumes/AD/Rima/RAFT_Project")
PDF_DIR = PROJECT_ROOT / "data" / "raw"
MD_DIR = PROJECT_ROOT / "data" / "markdown"

def main():
    MD_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs in {PDF_DIR}. Put 5–10 PDFs there.")
        return

    print(f"🚀 Converting {len(pdfs)} PDFs with Docling…")
    for pdf in pdfs:
        try:
            result = converter.convert(str(pdf))
            md_text = result.document.export_to_markdown()
            out_path = MD_DIR / f"{pdf.stem}.md"
            out_path.write_text(md_text, encoding="utf-8")
            print(f"✅ {pdf.name} → {out_path.name}")
        except Exception as e:
            print(f"❌ Failed {pdf.name}: {e}")

if __name__ == "__main__":
    main()
