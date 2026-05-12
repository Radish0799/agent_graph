import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE   = 2048
OVERLAP      = 128
MIN_PARA_LEN = CHUNK_SIZE // 2  # 段落短於此就往下合併

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    separators=[ 
                "="*60,
                "\n══ ", # 給Infobox用
                "。\n\n",
                "。", 
                "。\n",
                "！\n\n", 
                "！", 
                "？\n\n", 
                "？", 
                "!\n\n",
                "!", 
                "?\n\n", 
                "?",
                " ", 
                ""],
    keep_separator=True,
)

def _pre_merge(text: str, min_len: int = MIN_PARA_LEN) -> str:
    """
    把以雙換行分隔的段落做合併：
    若某段落長度 < min_len，就和下一段用雙換行拼接，
    直到累積長度 >= min_len 或沒有下一段為止。
    """
    # 正規化：3個以上換行 → 2個
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paras = text.split("\n\n")

    merged, buf = [], ""
    for para in paras:
        if buf:
            buf += "\n\n" + para
        else:
            buf = para
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""

    if buf:  # 剩餘尾段直接加入
        if merged:
            merged[-1] += "\n\n" + buf  # 和前一段合併，避免孤兒小 chunk
        else:
            merged.append(buf)

    return "\n\n".join(merged)


def tidy_skill(folder_name: str, source_file: str) -> str:
    try:
        if not os.path.exists(source_file):
            return f"[tidy_skill 錯誤] 來源檔案不存在：{source_file}"

        with open(source_file, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            return f"[tidy_skill 錯誤] 來源檔案是空的：{source_file}"

        os.makedirs(folder_name, exist_ok=True)

        merged    = _pre_merge(text)          # ← 關鍵：合併短段落
        chunks    = _splitter.split_text(merged)
        base_name = os.path.splitext(os.path.basename(source_file))[0]

        for idx, chunk in enumerate(chunks, start=1):
            out_path = os.path.join(folder_name, f"{base_name}_{idx}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(chunk)

        return f"已將結果儲存至 ./{folder_name} 資料夾內，共 {len(chunks)} 個 chunk"

    except Exception as e:
        return f"[tidy_skill 錯誤] {e}"


if __name__ == "__main__":
    tidy_skill("test", "rabbit.txt")