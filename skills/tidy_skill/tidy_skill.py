import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE   = 2048
OVERLAP      = 128
MIN_PARA_LEN = CHUNK_SIZE // 2

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    separators=[
        "="*60,
        "\n══ ",
        "。\n\n", "。", "。\n",
        "！\n\n", "！",
        "？\n\n", "？",
        "!\n\n", "!",
        "?\n\n", "?",
        " ", "",
    ],
    keep_separator=True,
)

def _pre_merge(text: str, min_len: int = MIN_PARA_LEN) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paras = text.split("\n\n")
    merged, buf = [], ""
    for para in paras:
        buf = buf + "\n\n" + para if buf else para
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += "\n\n" + buf
        else:
            merged.append(buf)
    return "\n\n".join(merged)

def _extract_wiki_title(text: str) -> str:
    """從文本開頭抓取 【Wikipedia：xxx】 標題行。找不到則回傳空字串。"""
    m = re.search(r"(【Wikipedia：[^】]+】)", text)
    return m.group(1) if m else ""

def tidy_skill(folder_name: str, source_file: str) -> str:
    try:
        if not os.path.exists(source_file):
            return f"[tidy_skill 錯誤] 來源檔案不存在：{source_file}"
        with open(source_file, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return f"[tidy_skill 錯誤] 來源檔案是空的：{source_file}"

        wiki_title = _extract_wiki_title(text)  # ← 抓標題

        os.makedirs(folder_name, exist_ok=True)
        merged = _pre_merge(text)
        chunks = _splitter.split_text(merged)

        base_name = os.path.splitext(os.path.basename(source_file))[0]
        for idx, chunk in enumerate(chunks, start=1):
            # ← 若標題不在 chunk 開頭，就補上
            if wiki_title and not chunk.lstrip().startswith(wiki_title):
                chunk = wiki_title + "\n" + chunk
            out_path = os.path.join(folder_name, f"{base_name}_{idx}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(chunk)

        return f"已將結果儲存至 ./{folder_name} 資料夾內，共 {len(chunks)} 個 chunk"
    except Exception as e:
        return f"[tidy_skill 錯誤] {e}"

if __name__ == "__main__":
    tidy_skill("test", "rabbit.txt")