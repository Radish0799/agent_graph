import os

def new_folder_skill(folder_name: str) -> str:
    try:
        if os.path.exists(folder_name):
            return f"已新增資料夾：{folder_name}"
        os.makedirs(folder_name, exist_ok=True)
        return f"已新增資料夾：{folder_name}"
    except Exception as e:
        return f"[new_folder_skill 錯誤] {e}"