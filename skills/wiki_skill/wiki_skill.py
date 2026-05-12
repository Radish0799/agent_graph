import opencc
_CC = opencc.OpenCC("t2s")

import unicodedata
import wikipedia
import requests
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────
# 工具：計算字串的「顯示寬度」（CJK 字元佔 2 格）
# ─────────────────────────────────────────────────────────

def _display_width(s: str) -> int:
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _pad(s: str, width: int) -> str:
    """左對齊填充，以顯示寬度為準。"""
    return s + " " * max(0, width - _display_width(s))


def _truncate(s: str, max_display_w: int) -> str:
    """依顯示寬度截斷，超出時加 '…'。"""
    w = 0
    result = []
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > max_display_w - 1:
            result.append("…")
            break
        result.append(ch)
        w += cw
    return "".join(result)


# ─────────────────────────────────────────────────────────
# Infobox：改為 key: value 清單格式
# ─────────────────────────────────────────────────────────
def _parse_single_infobox(infobox, index: int, total: int) -> str:
    """
    將單一 infobox BeautifulSoup 元素解析成 key: value 文字區塊。
    index / total 用來產生標題編號（total=1 時不顯示編號）。
    """
    caption_tag = infobox.find("caption")
    caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""

    entries = []
    for tr in infobox.find_all("tr"):
        ths = tr.find_all("th")
        tds = tr.find_all("td")

        if ths and not tds:
            text = " ".join(th.get_text(" ", strip=True) for th in ths)
            if text and _display_width(text) <= 40:
                entries.append(("section", text))
            continue

        if ths and tds:
            key = ths[0].get_text(" ", strip=True)
            val = " ".join(td.get_text(" ", strip=True) for td in tds)
            if not key or not val:
                continue
            if _display_width(key) > 30:
                continue
            entries.append(("kv", (key, val)))
            continue

    if not entries:
        return ""

    kv_keys = [v[0] for t, v in entries if t == "kv"]
    key_w = min(max((_display_width(k) for k in kv_keys), default=8), 20)

    num_tag = f"（{index}/{total}）" if total > 1 else ""
    title_line = f"══ Infobox{num_tag}：{caption} ══" if caption else f"══ Infobox{num_tag} ══"

    lines = [title_line]
    for kind, val in entries:
        if kind == "section":
            lines.append(f"\n  [{val}]")
        else:
            key, value = val
            value = _truncate(value.replace("\n", " "), 80)
            lines.append(f"  {_pad(key, key_w)} : {value}")

    return "\n".join(lines)


def _fetch_infobox(page_url: str) -> str:
    """
    解析頁面上所有 Infobox，多個時標示編號，以空行分隔。
    """
    try:
        resp = requests.get(page_url, timeout=10,
                            headers={"User-Agent": "wiki_skill/1.0"})
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    infoboxes = soup.find_all("table", class_=lambda c: c and "infobox" in c)
    if not infoboxes:
        return ""

    total = len(infoboxes)
    blocks = []
    for idx, ib in enumerate(infoboxes, 1):
        block = _parse_single_infobox(ib, idx, total)
        if block:
            blocks.append(block)

    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────
# wikitable：以正確 CJK 欄寬渲染
# ─────────────────────────────────────────────────────────

_MAX_CELL_W = 30   # 單一格顯示寬度上限（超出截斷）


def _fetch_html_tables(page_url: str) -> str:
    try:
        resp = requests.get(page_url, timeout=10,
                            headers={"User-Agent": "wiki_skill/1.0"})
        resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all(
        "table",
        class_=lambda c: c and "wikitable" in c and "infobox" not in c
    )
    if not tables:
        return ""

    output_parts = []
    for idx, table in enumerate(tables, 1):
        caption = table.find("caption")
        title = caption.get_text(" ", strip=True) if caption else f"表格 {idx}"

        # 建立 matrix（字串二維陣列）
        matrix = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [c.get_text(" ", strip=True).replace("\n", " ")
                   for c in cells]
            if any(row):
                matrix.append(row)

        if not matrix:
            continue

        # 統一欄數
        max_cols = max(len(r) for r in matrix)
        for r in matrix:
            while len(r) < max_cols:
                r.append("")

        # 截斷每格
        matrix = [
            [_truncate(cell, _MAX_CELL_W) for cell in row]
            for row in matrix
        ]

        # 計算各欄顯示寬度
        col_widths = []
        for ci in range(max_cols):
            w = max(_display_width(r[ci]) for r in matrix)
            col_widths.append(max(w, 4))

        def sep_line() -> str:
            return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        def row_line(row) -> str:
            parts = []
            for ci in range(max_cols):
                parts.append(" " + _pad(row[ci], col_widths[ci]) + " ")
            return "|" + "|".join(parts) + "|"

        lines = [f"\n  ── 表格：{title} ──", sep_line()]
        for ri, row in enumerate(matrix):
            lines.append(row_line(row))
            if ri == 0:          # 表頭後加分隔線
                lines.append(sep_line())
        lines.append(sep_line())
        output_parts.append("\n".join(lines))

    return "\n".join(output_parts)


# ─────────────────────────────────────────────────────────
# 主函式（介面與原版完全相同）
# ─────────────────────────────────────────────────────────

def wiki_skill(keyword: str, file_name: str, lang: str = "zh") -> str:
    """
    爬取 Wikipedia 頁面內容並存成結構化 txt 檔。

    Args:
        keyword:   搜尋關鍵字
        file_name: 輸出檔案路徑（.txt）
        lang:      Wikipedia 語言代碼（預設 "zh"）

    Returns:
        成功或錯誤訊息字串
    """
    wikipedia.set_lang(lang)
    try:
        results = wikipedia.search(keyword, results=5)
        if not results:
            return f"[wiki_skill 錯誤] 找不到「{keyword}」相關頁面。"

        page = None
        used_title = ""
        for candidate in results:
            try:
                page = wikipedia.page(candidate, auto_suggest=False)
                used_title = candidate
                break
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        used_title = e.options[0]
                        break
                    except Exception:
                        continue
            except Exception:
                continue

        if page is None:
            return f"[wiki_skill 錯誤] 無法取得「{keyword}」任何頁面內容。"

        # ── 抓取結構化資料 ───────────────────────────
        infobox_text = _fetch_infobox(page.url)
        tables_text  = _fetch_html_tables(page.url)

        # ── 組合輸出 ─────────────────────────────────
        SEP  = "=" * 60
        DASH = "─" * 60
        parts = [
            f"【Wikipedia：{used_title}】",
            f"URL：{page.url}",
            SEP,
        ]

        if infobox_text:
            parts += ["", infobox_text, ""]

        if tables_text:
            parts += [
                "",
                DASH,
                "【頁面表格彙整】",
                DASH,
                tables_text,
                "",
            ]

        parts += [
            SEP,
            "【正文內容】",
            SEP,
            "",
            page.content,
        ]

        full_text = "\n\n".join(parts)
        full_text = _CC.convert(full_text) # to簡體

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_text)

        return f"爬取與儲存成功，存成 {file_name}"

    except Exception as e:
        return f"[wiki_skill 錯誤] {e}"

# test
if __name__ == "__main__":
    print(wiki_skill("金色暗影", "rabbit.txt", lang="zh"))