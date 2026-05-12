import os


def load_txt_skill(file_path: str) -> str:
    try:
        if not os.path.exists(file_path):
            return f"[load_txt_skill 錯誤] 檔案不存在：{file_path}"
        if not os.path.isfile(file_path):
            return f"[load_txt_skill 錯誤] 路徑不是檔案：{file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return f"[load_txt_skill 警告] 檔案內容為空：{file_path}"

        return content

    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e2:
            return f"[load_txt_skill 錯誤] 編碼問題：{e2}"
    except Exception as e:
        return f"[load_txt_skill 錯誤] {e}"