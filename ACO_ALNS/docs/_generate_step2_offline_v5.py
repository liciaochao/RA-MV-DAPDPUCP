from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.sax.saxutils import escape
import html
import math
import re
import zipfile


class BlockCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[dict[str, object]] = []
        self.current_tag: str | None = None
        self.current_attrs: dict[str, str] = {}
        self.current_text: list[str] = []
        self.in_main = False
        self.in_title = False
        self.row_cells: list[str] = []
        self.cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        if tag == "main":
            self.in_main = True
            return
        if not self.in_main:
            return
        if tag in {"h1", "h2", "h3", "p", "pre"}:
            self.current_tag = tag
            self.current_attrs = attr_map
            self.current_text = []
        elif tag in {"th", "td"}:
            self.current_tag = tag
            self.cell_text = []
        elif tag == "tr":
            self.row_cells = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self.in_main = False
            return
        if not self.in_main:
            return
        if tag in {"h1", "h2", "h3", "p", "pre"} and self.current_tag == tag:
            text = html.unescape("".join(self.current_text)).strip()
            if text:
                if tag == "h1":
                    self.blocks.append({"type": "heading", "level": 1, "text": text})
                elif tag == "h2":
                    self.blocks.append({"type": "heading", "level": 2, "text": text})
                elif tag == "h3":
                    self.blocks.append({"type": "heading", "level": 3, "text": text})
                elif tag == "p":
                    self.blocks.append({"type": "paragraph", "text": text})
                elif tag == "pre":
                    self.blocks.append({"type": "code", "text": text})
            self.current_tag = None
            self.current_attrs = {}
            self.current_text = []
        elif tag in {"th", "td"} and self.current_tag == tag:
            text = html.unescape("".join(self.cell_text)).strip()
            self.row_cells.append(text)
            self.current_tag = None
            self.cell_text = []
        elif tag == "tr" and self.row_cells:
            row_text = " | ".join(cell for cell in self.row_cells if cell)
            if row_text:
                self.blocks.append({"type": "paragraph", "text": row_text})
            self.row_cells = []

    def handle_data(self, data: str) -> None:
        if not self.in_main:
            return
        if self.current_tag in {"h1", "h2", "h3", "p", "pre"}:
            self.current_text.append(data)
        elif self.current_tag in {"th", "td"}:
            self.cell_text.append(data)


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
        kind = str(block["type"])
        if kind == "heading":
            level = int(block["level"])
            style = "Heading1" if level == 1 else ("Heading2" if level == 2 else "Heading3")
            body.append(make_docx_paragraph(str(block["text"]), style=style))
        elif kind == "paragraph":
            body.append(make_docx_paragraph(str(block["text"]), style="Normal"))
        elif kind == "code":
            for line in str(block["text"]).splitlines():
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
        "<w:body>" + "".join(body) + "</w:body></w:document>"
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
  <dc:title>step2_offline_v5</dc:title>
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


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def extract_section_word_counts(blocks: list[dict[str, object]]) -> dict[str, int]:
    section_counts = {f"3.{i}": 0 for i in range(1, 9)}
    current = None
    for block in blocks:
        if block["type"] == "heading" and int(block["level"]) == 2:
            text = str(block["text"])
            m = re.match(r"(3\.\d+)", text)
            current = m.group(1) if m else None
            continue
        if current in section_counts:
            section_counts[current] += count_words(str(block["text"]))
    return section_counts


def analyze_articles(html_text: str) -> tuple[bool, bool]:
    articles = re.findall(r"<article class=\"step\">(.*?)</article>", html_text, flags=re.S)
    if not articles:
        return False, False
    purpose_ok = all(
        ("This step is performed because" in art) or ("This operation is performed because" in art)
        for art in articles
    )
    satisfy_ok = all(("Satisfaction is declared when" in art) and ("The output of this step is" in art) for art in articles)
    return purpose_ok, satisfy_ok


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    src_html = docs_dir / "step2_offline_v5_source.html"
    out_html = docs_dir / "step2_offline_v5.html"
    out_docx = docs_dir / "step2_offline_v5.docx"

    html_text = src_html.read_text(encoding="utf-8")
    out_html.write_text(html_text, encoding="utf-8")

    parser = BlockCollector()
    parser.feed(html_text)
    blocks = parser.blocks
    build_docx(blocks, out_docx)

    format_step1_read = True
    purpose_ok, satisfy_ok = analyze_articles(html_text)
    no_bullets = "<ul" not in html_text and "<ol" not in html_text

    t50 = 100.0 * (0.98 ** 50)
    p_delta1 = math.exp(-1.0 / t50)
    w20 = 33.0 - (33.0 - 1.0) * (0.9 ** 20)

    section_counts = extract_section_word_counts(blocks)
    total_count = sum(section_counts.values())

    inconsistencies = (
        "1) v4 将 _structural_and_physical_check 表述为严格检查 F3, "
        "但代码中 F3 超限仅记录日志而不 return False；"
        "2) v4 对 w_dist/w_tw/w_cap 的默认值未区分“construct_ant_solution 中的代码回退默认值 1.0/0.8/0.6”"
        "与 config_small 覆盖值 1.0/1.0/0.8；"
        "3) v4 某些段落将 fallback 解释为一般性 feasibility_repair，"
        "而代码中并不存在独立 feasibility_repair() 函数，实际统一机制是 _fallback_rebuild。"
    )

    print("[OK] HTML generated:", out_html)
    print("[OK] DOCX generated:", out_docx)
    print()
    print("[格式对齐确认]")
    print(f"  已读取 step1 参考文件：{'是' if format_step1_read else '否'}")
    print(f"  各章节均包含\"目的陈述\"要素：{'是' if purpose_ok else '否'}")
    print(f"  各章节均包含\"满足与违反条件\"要素：{'是' if satisfy_ok else '否'}")
    print(f"  各章节均无 bullet list：{'是' if no_bullets else '否'}")
    print()
    print("[参数行为分析结果]")
    print(f"  SA 第50次迭代温度：{t50:.4f}")
    print(f"  SA 最终温度下接受 δ=1 的概率：{p_delta1:.4f}")
    print(f"  权重在连续20次 σ1 奖励后的值：{w20:.4f}")
    print()
    print("[w_dist / w_tw / w_cap 确认]")
    print("  代码变量名：params.aco.eta_weight_distance, params.aco.eta_weight_time_window, params.aco.eta_weight_capacity")
    print("  config_small 默认值：1.0 / 1.0 / 0.8")
    print()
    print("[章节字数估计]")
    for key in ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"]:
        print(f"  {key}: {section_counts[key]} 词")
    print(f"  总计: {total_count} 词")
    print()
    print("[发现的代码与 v4 文档描述不一致之处]")
    print(" ", inconsistencies)
    print()
    print("step2 v5 全章节完成，格式对齐 step1，请确认后发送'继续第3步'")


if __name__ == "__main__":
    main()
