from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


def parse_markdown(md_text: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    lines = md_text.splitlines()

    in_code = False
    code_lines: list[str] = []
    para_lines: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        nonlocal para_lines
        if para_lines:
            text = " ".join(part.strip() for part in para_lines if part.strip())
            if text:
                blocks.append({"type": "paragraph", "text": text})
        para_lines = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append({"type": "list", "items": list_items[:]})
        list_items = []

    for raw in lines + [""]:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                blocks.append({"type": "code", "text": "\n".join(code_lines)})
                in_code = False
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 1, "text": stripped[2:].strip()})
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 2, "text": stripped[3:].strip()})
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 3, "text": stripped[4:].strip()})
            continue
        if stripped.startswith("#### "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 4, "text": stripped[5:].strip()})
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
            continue

        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == ".":
            flush_paragraph()
            list_items.append(stripped)
            continue

        para_lines.append(line)

    return blocks


def render_inline_html(text: str) -> str:
    parts = text.split("`")
    out: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            out.append(escape(part))
        else:
            out.append(f"<code>{escape(part)}</code>")
    return "".join(out)


def render_html(blocks: list[dict[str, object]]) -> str:
    body: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            level = int(block["level"])
            level = min(max(level, 1), 4)
            body.append(f"<h{level}>{render_inline_html(str(block['text']))}</h{level}>")
        elif kind == "paragraph":
            body.append(f"<p>{render_inline_html(str(block['text']))}</p>")
        elif kind == "list":
            body.append("<ul>")
            for item in block["items"]:  # type: ignore[index]
                body.append(f"  <li>{render_inline_html(str(item))}</li>")
            body.append("</ul>")
        elif kind == "code":
            body.append(f"<pre><code>{escape(str(block['text']))}</code></pre>")

    content = "\n".join(body)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2 Offline v4</title>
<style>
body {{
  margin: 0;
  background: #f4f6f8;
  color: #1f2933;
  font-family: "Times New Roman", Georgia, "Noto Serif SC", serif;
  line-height: 1.75;
}}
main {{
  max-width: 1180px;
  margin: 28px auto;
  background: #ffffff;
  border: 1px solid #d9e2ec;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  padding: 40px 52px;
}}
h1, h2, h3, h4 {{
  color: #102a43;
  line-height: 1.35;
}}
h1 {{
  font-size: 2rem;
  border-bottom: 3px solid #d9e2ec;
  padding-bottom: 0.55rem;
}}
h2 {{ margin-top: 2rem; font-size: 1.5rem; }}
h3 {{ margin-top: 1.25rem; font-size: 1.22rem; }}
h4 {{ margin-top: 1rem; font-size: 1.05rem; }}
p {{
  margin: 0.5rem 0 0.95rem 0;
  text-align: justify;
}}
ul {{
  margin: 0.35rem 0 0.95rem 0;
  padding-left: 1.4rem;
}}
li {{ margin: 0.22rem 0; }}
code {{
  background: #eef2f7;
  border-radius: 4px;
  padding: 1px 5px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.93em;
}}
pre {{
  white-space: pre-wrap;
  background: #f7fafc;
  border: 1px solid #cbd5e0;
  border-left: 4px solid #486581;
  padding: 12px 14px;
  margin: 0.9rem 0 1.1rem 0;
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.91rem;
  line-height: 1.55;
}}
</style>
</head>
<body>
<main>
{content}
</main>
</body>
</html>
"""


def strip_inline_markers(text: str) -> str:
    return text.replace("`", "")


def make_docx_paragraph(text: str, style: str = "Normal", preserve: bool = False) -> str:
    if text == "":
        return "<w:p/>"
    style_xml = f"<w:pPr><w:pStyle w:val=\"{escape(style)}\"/></w:pPr>"
    space_attr = " xml:space=\"preserve\"" if preserve else ""
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        f"<w:t{space_attr}>{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def build_docx(blocks: list[dict[str, object]], out_path: Path) -> None:
    body: list[str] = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            level = int(block["level"])
            level = min(max(level, 1), 3)
            style = f"Heading{level}"
            body.append(make_docx_paragraph(strip_inline_markers(str(block["text"])), style=style))
        elif kind == "paragraph":
            body.append(make_docx_paragraph(strip_inline_markers(str(block["text"])), style="Normal"))
        elif kind == "list":
            for item in block["items"]:  # type: ignore[index]
                body.append(make_docx_paragraph("• " + strip_inline_markers(str(item)), style="Normal"))
        elif kind == "code":
            code_text = str(block["text"])
            if code_text == "":
                body.append(make_docx_paragraph("", style="Code"))
            else:
                for line in code_text.splitlines():
                    body.append(make_docx_paragraph(line, style="Code", preserve=True))
        body.append(make_docx_paragraph(""))

    body.append(
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        + "".join(body)
        + "</w:body></w:document>"
    )

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="20"/>
    </w:rPr>
  </w:style>
</w:styles>
"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
"""

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>step2_offline_v4</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)


def main() -> None:
    base = Path(__file__).resolve().parent
    src = base / "step2_offline_v4_source.md"
    html_path = base / "step2_offline_v4.html"
    docx_path = base / "step2_offline_v4.docx"

    md_text = src.read_text(encoding="utf-8")
    blocks = parse_markdown(md_text)
    html_text = render_html(blocks)
    html_path.write_text(html_text, encoding="utf-8")
    build_docx(blocks, docx_path)

    print(f"[OK] HTML generated: {html_path}")
    print(f"[OK] DOCX generated: {docx_path}")


if __name__ == "__main__":
    main()
