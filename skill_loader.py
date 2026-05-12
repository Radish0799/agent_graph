"""
skill_loader.py
===============
動態掃描 ./skills/ 目錄，載入所有符合規範的 skill。

Skill 資料夾規範：
  skills/
  └── <skill_name>/
      ├── skill_info.json     ← 說明文件（必須）
      └── <skill_name>.py     ← 實作（必須，函數名稱需與資料夾名稱相同）

載入後提供：
  SKILL_REGISTRY    : dict[str, Callable]   → 供 ExecuteAgent 呼叫
  SKILL_DESCRIPTIONS: str                   → 供 SuperviseAgent 規劃時參考
"""

import os
import json
import importlib.util
from typing import Callable
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── 設定 skills 根目錄 ─────────────────────────────────────────
_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _json_to_description(info: dict) -> str:
    """
    將 skill_info.json 的內容轉換成給 LLM 看的自然語言說明。
    """
    lines = []
    lines.append(f"skill_name: {info.get('skill_name', '?')}")
    lines.append(f"  描述: {info.get('description', '')}")

    inputs = info.get("input", [])
    if inputs:
        lines.append("  input 參數:")
        for i, p in enumerate(inputs):
            lines.append(
                f"    [{i}] {p.get('name', '')} ({p.get('type', 'str')}): {p.get('description', '')}"
            )

    lines.append(f"  回傳: {info.get('returns', '')}")

    example = info.get("example")
    if example:
        lines.append(f"  範例 input: {example.get('input', [])}")
        lines.append(f"  範例回傳: {example.get('returns', '')}")

    return "\n".join(lines)


def load_skills(skills_dir: str = _SKILLS_DIR) -> tuple[dict[str, Callable], str]:
    """
    掃描 skills_dir 目錄，動態載入所有 skill。

    Args:
        skills_dir: skill 根目錄路徑

    Returns:
        (SKILL_REGISTRY, SKILL_DESCRIPTIONS)
        - SKILL_REGISTRY     : {skill_name: callable}
        - SKILL_DESCRIPTIONS : 所有 skill 的說明文字（給 LLM 用）
    """
    registry:     dict[str, Callable] = {}
    descriptions: list[str]           = []
    load_errors:  list[str]           = []

    if not os.path.isdir(skills_dir):
        console.print(f"[bold red][SkillLoader] skills 目錄不存在：{skills_dir}[/bold red]")
        return registry, ""

    for entry in sorted(os.scandir(skills_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        skill_name = entry.name
        skill_dir  = entry.path

        # ── 1. 找 skill_info.json ──────────────────────────────
        info_path = os.path.join(skill_dir, "skill_info.json")
        if not os.path.isfile(info_path):
            load_errors.append(f"  ⚠ [{skill_name}] 缺少 skill_info.json，跳過")
            continue

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except json.JSONDecodeError as e:
            load_errors.append(f"  ⚠ [{skill_name}] skill_info.json 格式錯誤：{e}，跳過")
            continue

        # ── 2. 找 <skill_name>.py ──────────────────────────────
        py_path = os.path.join(skill_dir, f"{skill_name}.py")
        if not os.path.isfile(py_path):
            load_errors.append(f"  ⚠ [{skill_name}] 缺少 {skill_name}.py，跳過")
            continue

        # ── 3. 動態 import ─────────────────────────────────────
        try:
            spec   = importlib.util.spec_from_file_location(f"skills.{skill_name}", py_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            load_errors.append(f"  ⚠ [{skill_name}] import 失敗：{e}，跳過")
            continue

        # ── 4. 取得函數（函數名稱需與 skill_name 相同）──────────
        fn = getattr(module, skill_name, None)
        if fn is None or not callable(fn):
            load_errors.append(
                f"  ⚠ [{skill_name}] {skill_name}.py 內找不到函數 {skill_name}()，跳過"
            )
            continue

        # ── 5. 登記 ────────────────────────────────────────────
        registry[skill_name] = fn
        descriptions.append(_json_to_description(info))

    # ── 印出載入結果 ────────────────────────────────────────────
    _print_load_summary(registry, load_errors)

    skill_descriptions = (
        "可用 Skill 清單（每個 step 的 skill_name 必須從這裡選）：\n\n"
        + "\n\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))
    )

    return registry, skill_descriptions


def _print_load_summary(registry: dict, errors: list[str]) -> None:
    table = Table(title="[bold]Skill 載入結果[/bold]", box=box.SIMPLE_HEAVY)
    table.add_column("Skill",   style="cyan",  width=22)
    table.add_column("狀態",    style="bold",  width=10)
    table.add_column("函數",    style="dim")

    for name, fn in registry.items():
        table.add_row(name, "[green]✅ 已載入[/green]", str(fn))

    console.print(table)

    for err in errors:
        console.print(f"[yellow]{err}[/yellow]")

    console.print(
        f"  共載入 [bold green]{len(registry)}[/bold green] 個 skill，"
        f"[bold yellow]{len(errors)}[/bold yellow] 個警告\n"
    )