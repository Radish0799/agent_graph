import json
import re
import os
import glob
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from llama_cpp import Llama

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from skill_loader import load_skills

MAX_PLAN_ITER  = 10
MAX_EVAL_RETRY = 3
MAX_ACE_RETRY  = 2

console = Console()


class Node(str, Enum):
    PLAN      = "PLAN"
    SUPERVISE = "SUPERVISE"
    DISPATCH  = "DISPATCH"
    EXECUTE   = "EXECUTE"
    EVAL      = "EVAL"
    SUMMARY   = "SUMMARY"
    END       = "END"


@dataclass
class GraphState:
    user_task:           str        = ""
    sub_context:         str        = ""
    depth:               int        = 0
    current_plan:        str        = ""
    supervisor_feedback: str        = ""
    plan_iterations:     int        = 0
    workflow:            list[dict] = field(default_factory=list)
    current_step_index:  int        = 0
    _dispatched:         bool       = False
    step_results:        list[dict] = field(default_factory=list)
    current_output:      str        = ""
    eval_feedback:       str        = ""
    eval_retry_count:    int        = 0
    failed_keywords:     list[str]  = field(default_factory=list)
    replan_count:        int        = 0
    final_output:        str                = ""
    execution_log:       list[dict] = field(default_factory=list)

    def log(self, node: Node, action: str, detail: str = "") -> None:
        self.execution_log.append({
            "node": node.value, "action": action, "detail": detail[:120],
        })

    @property
    def current_step(self) -> dict | None:
        if self.current_step_index < len(self.workflow):
            return self.workflow[self.current_step_index]
        return None


class LLMClient:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 16384):
        self._model = Llama(
            model_path, verbose=False,
            n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
        )

    def generate(self, messages: list[dict], label: str = "LLM",
                 max_tokens: int = 4096, temperature: float = 0.0) -> str:
        console.print(Rule(f"[bold cyan]► {label} — INPUT[/bold cyan]", style="cyan"))
        styles = {
            "system":    ("[yellow]SYSTEM[/yellow]",                "yellow"),
            "assistant": ("[magenta]ASSISTANT (history)[/magenta]", "magenta"),
            "user":      ("[green]USER[/green]",                    "green"),
        }
        for msg in messages:
            title, border = styles.get(msg["role"], ("[white]MSG[/white]", "white"))
            console.print(Panel(msg["content"], title=title, border_style=border, padding=(0, 1)))

        output: str = self._model.create_chat_completion(
            messages=messages,
            stop=["<|im_end|>", "<|endoftext|>"],
            max_tokens=max_tokens,
            temperature=temperature,
        )["choices"][0]["message"]["content"]

        console.print(Rule(f"[bold magenta]◄ {label} — OUTPUT[/bold magenta]", style="magenta"))
        console.print(Panel(output, title="[magenta]ASSISTANT[/magenta]",
                            border_style="magenta", padding=(0, 1)))
        console.print()
        return output


def _sort_key(path: str) -> tuple:
    nums = re.findall(r"\d+", os.path.basename(path))
    return tuple(int(n) for n in nums) if nums else (0,)


def re_process_agent(question: str, chunks_folder: str, output_file: str, llm: LLMClient) -> str:
    if not os.path.exists(chunks_folder):
        return f"[re_process_agent 错误] 文件夹不存在：{chunks_folder}"

    chunk_files = sorted(
        glob.glob(os.path.join(chunks_folder, "*.txt")),
        key=_sort_key,
    )
    if not chunk_files:
        return f"[re_process_agent 错误] 文件夹内没有 .txt 档案：{chunks_folder}"

    system_prompt = """
你是一位严谨的知识萃取助手，负责维护一份持续更新的「重点 buffer」。

任务：
1. 注意：文章中的资讯不一定与目标问题相关。
2. 根据问题，从「文献」中找出相关资讯。
3. 将新发现与「已知重点」合并，移除不相关或重复的内容。
4. 若发现新的别名/别称/简称关系，记录至 aliases。
5. buffer 为总结当前所获得的资讯，上限约 800 字，若无重点则留白。
6.「已知重点」中的内容也属于正确资讯。

规则：
（A）必须同时输出 buffer 与 aliases 两个区块
（B）buffer 保留可能与问题相关的资讯，若资讯须更新，可更新buffer的资讯
（C）禁止捏造文章内没有的资讯
（D）若buffer无须更新，buffer 维持原样输出即可
（E）aliases 只记录本轮新发现的别名

输出格式：
```buffer
更新后的重点整理（上限 800 字），必须是事实
```

若本段无相关资讯，留白（不填入任何文字）：
```buffer

```

注意：alias 与 canonical 必须指向同一人/事/物，例如："苹果" 可称作 "apple" 的时候
```aliases
[{"alias": "A的名称", "canonical": "A的别名或绰号等"}, {"alias": "苹果", "canonical": "apple"}, ...]
```
若无新别名：
```aliases
[]
```
"""

    eval_system = """
你是一位严格的资讯核查员。

核查方式：
对 buffer 中每一条新资讯，你必须在「原文文献」中明确指出对应的原文句子。
aliases 中的每一组 alias/canonical 关系，同样必须能在原文文献中找到明确依据。

（A）找得到对应原文：该条通过。
（B）找不到对应原文：该条视为捏造，列入 issues。

注意：「找不到冲突」不等于通过，必须「找得到来源」才算通过。
措辞改写允许，意思正确即可。
buffer 为空则直接通过。

输出格式（先逐条举证，再输出结论）：
```eval
{"passed": false, "issues": ["文献中没有指出牛顿喜欢吃苹果","问题描述2" ...]}
```

范例：
需要整理的目标是：牛顿喜不喜欢吃苹果
【原文文献（Newton.txt）】：
「
在《原理》一书中，提出牛顿运动定律与万有引力定律，此后数世纪间成为主导自然哲学与物理学的核心理论，直至后来被相对论部分取代。
」
【整理后的完整內容】：

【本轮新发现的 aliases】：
[]

整理后的完整内容属实（因没有资讯所以为空），通过，输出
```eval
{"passed": true, "issues": [""]}
```
"""

    ace_buffer: str = ""
    known_aliases: list[dict] = []

    for chunk_path in chunk_files:
        chunk_name = os.path.basename(chunk_path)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read().strip()
        if not chunk_text:
            continue

        alias_hint = ""
        if known_aliases:
            alias_hint = "\n已知別名表：\n" + "\n".join(
                f"  - 「{a['alias']}」也称作「{a['canonical']}」"
                for a in known_aliases
            )

        prev_buffer = ace_buffer
        feedback = ""
        accumulated_issues: list[str] = []

        for attempt in range(1, MAX_ACE_RETRY + 2):
            ace_user = f"""
需要整理的目标是：{question}

已知重点（已拥有的正确资讯）：
{prev_buffer if prev_buffer else '（目前为空）'}

文献：
「
{chunk_text}

{alias_hint}
」
"""
            if feedback:
                ace_user += f"\n\n【提示，请注意这部分问题】：\n{feedback}"

            raw = llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": ace_user},
                ],
                label=f"ACE [{chunk_name}] 嘗試#{attempt}",
                max_tokens=1024,
                temperature=0.0,
            ).strip()

            clean = _strip_think(raw)

            buf_match = re.search(r"```buffer\s*(.*?)```", clean, re.DOTALL | re.IGNORECASE)
            candidate_buffer = buf_match.group(1).strip() if buf_match else prev_buffer

            alias_match = re.search(r"```aliases\s*(.*?)```", clean, re.DOTALL | re.IGNORECASE)
            candidate_aliases: list[dict] = []
            if alias_match:
                try:
                    new_aliases = json.loads(alias_match.group(1).strip())
                    if isinstance(new_aliases, list):
                        candidate_aliases = [
                            e for e in new_aliases
                            if isinstance(e, dict) and "alias" in e and "canonical" in e
                            and not any(
                                a["alias"] == e["alias"] and a["canonical"] == e["canonical"]
                                for a in known_aliases
                            )
                        ]
                except json.JSONDecodeError:
                    pass

            eval_user = f"""
需要整理的目标是：{question}

【原文文献（{chunk_name}）】：
「
{prev_buffer if prev_buffer else ''}

{chunk_text}
」

【整理后的完整內容】：
{candidate_buffer}

【本轮新发现的 aliases】：
{json.dumps(candidate_aliases, ensure_ascii=False) if candidate_aliases else "[]"}
"""

            eval_raw = llm.generate(
                messages=[
                    {"role": "system", "content": eval_system},
                    {"role": "user",   "content": eval_user},
                ],
                label=f"Eval [{chunk_name}] 嘗試#{attempt}",
                max_tokens=512,
                temperature=0.0,
            ).strip()

            eval_clean = _strip_think(eval_raw)
            eval_match = re.search(r"```eval\s*(.*?)```", eval_clean, re.DOTALL | re.IGNORECASE)
            eval_parsed = None
            if eval_match:
                try:
                    eval_parsed = json.loads(eval_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

            passed = bool(eval_parsed.get("passed", False)) if eval_parsed else False
            issues = eval_parsed.get("issues", []) if eval_parsed else ["eval 解析失败"]

            if passed:
                ace_buffer = candidate_buffer
                known_aliases.extend(candidate_aliases)
                console.print(f"  [green]Eval 通過 [{chunk_name}] 嘗試#{attempt}[/green]")
                break
            else:
                for issue in issues:
                    if issue not in accumulated_issues:
                        accumulated_issues.append(issue)
                feedback = "\n".join(f"{i+1}.{item}" for i, item in enumerate(accumulated_issues))
                console.print(f"  [red]Eval 不通過 [{chunk_name}] 嘗試#{attempt}：{feedback}[/red]")
                if attempt > MAX_ACE_RETRY:
                    ace_buffer = prev_buffer
                    console.print(f"  [yellow]超出重重試上限，跳過{chunk_name}，buffer 回朔[/yellow]")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 针对问题：{question}\n\n")
        f.write(ace_buffer)
        if known_aliases:
            f.write("\n\n---\n## 补充资讯：\n")
            for a in known_aliases:
                f.write(f"- 「{a['alias']}」也可以称为「{a['canonical']}」\n")

    alias_summary = "、".join(
        f"{a['alias']}={a['canonical']}" for a in known_aliases
    ) if known_aliases else "（无）"

    return (
        f"已将重点整理储存至 {output_file}，"
        f"共处理 {len(chunk_files)} 个 chunk。"
        f"别名：{alias_summary}"
    )


# re_process_agent 的說明文字，给 SuperviseAgent 看的
_RE_PROCESS_DESCRIPTION = """
0. skill_name: re_process_agent
  描述: 循环读取指定文件夹内的每个 .txt chunk，针对用户问题萃取重点，累积写入目标档案。每次只处理一个 chunk，不会有 context 爆炸问题。
若某 chunk 与问题无关则自动跳过。 
  input 参数:
    [0] question (str): 用户核心问题，用来引导每个 chunk 的重点萃取，例如："芙莉莲的角色背景与剧情是什么？"
    [1] chunks_folder (str): 存放 chunk .txt 的文件夹路径，例如："./frieren_chunks"
    [2] output_file (str): 储存重点整理结果的目标档案名称，例如："frieren_summary.txt"
  回传: "已将重点整理储存至 <output_file>，共处理 <n> 个 chunk，有内容的 chunk 共 <m> 个"
  范例 input: ["芙莉莲的角色背景与剧情是什么？", "./frieren_chunks", "frieren_summary.txt"]
  范例回传: "已将重点整理储存至 frieren_summary.txt，共处理 42 个 chunk，有内容的 chunk 共 28 个"
"""

_SUB_WORKFLOW_DESCRIPTION = """
1. skill_name: sub_workflow
  描述: 读取指定档案作为已知资讯，引用前段的结果继续搜索，并将最终结果存成指定档案。
  input 参数:
    [0] context_file (str): 包含已知资讯的 .txt 档案路径，例如："largest_city_summary.txt"
    [1] output_file (str): 子流程最终结果要储存的档案名称，例如："city_population.txt"
  回传: 子 workflow 的最终输出字串
  使用时机: 当某步骤的结果需要作为下一阶段搜索的依据时
  范例 input: ["largest_city_summary.txt", "city_population.txt"]
  范例回传: "这座城市里面有分好几个区块，其中..."
"""
class BaseAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, state: GraphState) -> GraphState:
        raise NotImplementedError


class PlanAgent(BaseAgent):
    def __init__(self, llm: LLMClient, skill_descriptions: str):
        super().__init__(llm)
        self._skill_descriptions = skill_descriptions

    def run(self, state: GraphState) -> GraphState:
        console.print(Rule(f"[bold yellow]PLAN（第 {state.plan_iterations + 1} 輪）[/bold yellow]"))
        all_descriptions = "agent类型skill：\n"+ _RE_PROCESS_DESCRIPTION + _SUB_WORKFLOW_DESCRIPTION + "\n\n一般skill" + self._skill_descriptions
        system = f"""
你是一位操作电脑的工程师。
根据用户任务，规划在wiki搜索时的步骤，先抓出合理的关键字，再进行搜索，提供清晰的任务执行思路（条列式）。
若有主管反馈，请依反馈修正规划。

范例（用户任务：世界上最大的城市是哪座？该城市的人口分布是怎么样的）：
1. 使用 wiki_skill 搜索关键字"最大的城市"，存成档案largest_city.txt
2. 新增 chunks 文件夹 largest_city_chunk
3. 使用 tidy_skill 整理 largest_city.txt 资讯避免过长
4. 萃取城市名称
5. 用已知城市名称配合 sub_workflow 继续搜索其人口分布，并存成 city_population_summary.txt
6. 读取最终整理好的人口资讯 city_population_summary.txt

{"使用搜索时以下关键字已搜索过但无结果，禁止重复使用：\n" + ", ".join(state.failed_keywords) + "\n请务必使用不同关键字。" if state.failed_keywords else "" }

可使用的工具：
{all_descriptions}
"""
        user_content = f"用户任务：{state.user_task}"
        
        # 若有上層傳入的已知資訊，加入 prompt
        if state.sub_context:
            user_content += f"\n\n请根据以下已知信息进行规划：\n{state.sub_context}"
            
        if state.supervisor_feedback:
            user_content += f"\n\n主管反馈（请依此修正）：\n{state.supervisor_feedback}"

        state.current_plan = self.llm.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            label="規劃Agent",
        )
        state.plan_iterations += 1
        state.log(Node.PLAN, f"規劃第 {state.plan_iterations}輪", state.current_plan[:80])
        return state


class SuperviseAgent(BaseAgent):
    def __init__(self, llm: LLMClient, skill_descriptions: str):
        super().__init__(llm)
        self._skill_descriptions = skill_descriptions

    def run(self, state: GraphState) -> GraphState:
        console.print(Rule("[bold yellow]SUPERVISE[/bold yellow]"))

        all_descriptions = ""+ _RE_PROCESS_DESCRIPTION + _SUB_WORKFLOW_DESCRIPTION + "\n\n功能型skill" + self._skill_descriptions

        system = f"""
你是一位谨慎的主管，负责审查规划，先找出是否有不合逻辑的地方，并合理运用skill并输出具体的执行步骤清单。
Skill 清单
{all_descriptions}

审查规则：
(A) 若规划不够好、缺少步骤，只输出纯文字修改建议，不要 JSON。
(B) 若规划够好，开始安排步骤，输出以下格式的 JSON（不加其他文字）：

注意：
{"使用搜索时以下关键字已搜索过但无结果，禁止重复使用：\n" + ", ".join(state.failed_keywords) + "\n请务必使用不同关键字。" if state.failed_keywords else "" }

流程规则：
0. 每一步骤都会被直接执行，填入的字串确保不是引用
1. input 必须是字串阵列
2. 在范例的情况下，step 5使用的需要是sub_workflow，而不是使用wiki_skill
3. 每个 step 的 skill_name 必须完全符合上方 Skill 清单的名称
4. 读取txt档案时，先确保读取的是经过处理后的档案，避免长度过长
5. 注意档案命名，确保读取的档案名称与当初建档时命名相同
6. 最后，确保读取整理好的资料才算完整的流程


范例(A) 
wiki_skill 输入应该为简短的关键字而非完整问题，且第二段搜索依赖第一段结果时，应使用 sub_workflow。

范例(B)（假设有 n 个 step）：
```json
{{"approved": true, "workflow": [
{{"step": 1, "skill_name": "wiki_skill", "description": "在wiki上搜索关键字", "input": ["世界最大城市", "largest_city.txt"]}},
{{"step": 2, "skill_name": "new_folder_skill", "description": "新增 chunks 文件夹", "input": ["largest_city_chunks"]}},
{{"step": 3, "skill_name": "tidy_skill", "description": "整理资讯避免过长", "input": ["largest_city_chunks", "largest_city.txt"]}},
{{"step": 4, "skill_name": "re_process_agent", "description": "萃取城市名称", "input": ["世界上最大的城市叫什么名字？", "./largest_city_chunks", "largest_city_summary.txt"]}},
{{"step": 5, "skill_name": "sub_workflow", "description": "用已知城市名称继续搜索其人口分布", "input": ["largest_city_summary.txt", "city_population_summary.txt"]}},
{{"step": 6, "skill_name": "load_txt_skill", "description": "读取最终整理好的人口资讯", "input": ["city_population_summary.txt"]}}
]}}
"""

        user_content = f"用户任务：{state.user_task}\n\n规划思路：\n{state.current_plan}"
        if state.sub_context:
            user_content += f"\n\n已知信息：\n{state.sub_context}"
        
        raw = self.llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            label="主管Agent",
        )
        parsed = _extract_json(raw)

        if parsed and parsed.get("approved") is True:
            state.workflow            = parsed.get("workflow", [])
            state.supervisor_feedback = ""
            state.log(Node.SUPERVISE, "核准", f"共 {len(state.workflow)} 步骤")
        else:
            state.supervisor_feedback = raw
            state.log(Node.SUPERVISE, "退回", raw[:80])

            if state.plan_iterations >= MAX_PLAN_ITER:
                console.print(f"[bold red]規劃超出 {MAX_PLAN_ITER} 次上限，強制 fallback。[/bold red]")
                state.workflow = [
                    {"step": 1, "skill_name": "wiki_skill",
                     "description": "搜索主题", "input": [state.user_task, "output.txt"]},
                    {"step": 2, "skill_name": "load_txt_skill",
                     "description": "读取结果", "input": ["output.txt"]},
                ]
                state.supervisor_feedback = ""

        return state


class DispatchAgent(BaseAgent):
    def run(self, state: GraphState) -> GraphState:
        console.print(Rule("[bold yellow]DISPATCH[/bold yellow]"))

        if not state._dispatched:
            state.current_step_index = 0
            state._dispatched        = True
            console.print(f"  建立任務：共 [bold]{len(state.workflow)}[/bold] 步")
        else:
            state.current_step_index += 1
            state.eval_retry_count    = 0
            state.eval_feedback       = ""

        table = Table(title="執行步驟狀態", box=box.SIMPLE_HEAVY)
        table.add_column("#",    style="dim",  width=4)
        table.add_column("Skill", style="cyan", width=20)
        table.add_column("描述",  style="white")
        table.add_column("狀態",  style="bold")
        for s in state.workflow:
            i = s["step"] - 1
            if i < state.current_step_index:
                status = "[green]done[/green]"
            elif i == state.current_step_index:
                status = "[yellow]running[/yellow]"
            else:
                status = "[dim]pending[/dim]"
            table.add_row(str(s["step"]), s["skill_name"], s["description"], status)
        console.print(table)

        state.log(Node.DISPATCH, f"index={state.current_step_index}/{len(state.workflow)}")
        return state


class ExecuteAgent(BaseAgent):
    def __init__(self, llm: LLMClient, skill_registry: dict, skills_dir: str = "./skills"):
        super().__init__(llm)
        self._registry = skill_registry
        self._skills_dir = skills_dir

    def run(self, state: GraphState) -> GraphState:
        step    = state.current_step
        skill   = step["skill_name"]
        inputs  = step.get("input", [])
        attempt = state.eval_retry_count + 1

        console.print(Rule(f"[bold yellow]EXECUTE [{skill}] 嘗試 #{attempt}[/bold yellow]"))

        try:
            if skill == "re_process_agent":
                inputs = list(inputs)
                inputs[0] = state.user_task
                result = re_process_agent(*inputs, llm=self.llm)
            
            elif skill == "sub_workflow":
                if state.depth >= 2:
                    result = "[sub_workflow 跳過] 已達最大層數限制（2層）"
                else:
                    context_file = inputs[0] if len(inputs) > 0 else ""
                    output_file  = inputs[1] if len(inputs) > 1 else ""

                    sub_context = ""
                    if context_file and os.path.exists(context_file):
                        with open(context_file, "r", encoding="utf-8") as f:
                            sub_context = f.read().strip()
                    else:
                        sub_context = f"（找不到檔案：{context_file}）"

                    # 附加輸出檔名指示到 user_task
                    sub_task = state.user_task
                    if output_file:
                        sub_task += f"\n\n注意：请将最终整理好的资讯存成档案「{output_file}」，供后续流程读取。"

                    sub_graph = AgentGraph(
                        llm=self.llm,
                        skills_dir=self._skills_dir,
                        depth=state.depth + 1,
                    )
                    result = sub_graph.run(
                        user_task=sub_task,
                        sub_context=sub_context,
                    )
            
            elif skill in self._registry:
                result = self._registry[skill](*inputs)
            else:
                result = f"[错误] 找不到 skill：{skill}"

        except Exception as e:
            result = f"[Skill 执行错误] {skill}: {e}"

        state.current_output = str(result)
        state.log(Node.EXECUTE, f"[{skill}] 尝试#{attempt}", state.current_output[:80])
        console.print(Panel(state.current_output, title=f"[cyan]{skill} 回傳[/cyan]", border_style="cyan"))
        return state


class EvalAgent(BaseAgent):
    def run(self, state: GraphState) -> GraphState:
        console.print(Rule("[bold yellow]EVAL[/bold yellow]"))
        step = state.current_step

        system = """
你是一位负责评估的助手。

你的任务：
1. 先根据任务描述、skill 功能、输入内容、执行结果进行完整推理分析。
2. 判断结果是否符合预期。
3. 若不通过，指出具体原因。
4. 最终只输出 markdown JSON 区块。

评估标准：
- 若是读资料，只要读取成功就通过
- 无明显错误讯息
- 回传值合理

特别注意：
- 若回传值以 [错误] 或 [*_skill 错误] 开头，直接视为不通过。

输出格式必须严格遵守：

```json
{"passed": true/false, "feedback": "不通过原因；通过则空字串"}
```

规则：
* 必须使用 ```json``` 格式
* 必须先推理再输出最终结果
"""

        raw = self.llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"任务描述：{step['description']}\n"
                    f"skill 名称：{step['skill_name']}\n"
                    f"skill 输入：{step.get('input', [])}\n\n"
                    f"执行结果：\n{state.current_output}"
                )},
            ],
            label="Eval Agent",
        )

        clean_raw = _strip_think(raw)
        match = re.search(r"```json\s*(.*?)```", clean_raw, re.DOTALL | re.IGNORECASE)
        extracted = match.group(1).strip() if match else ""
        parsed = _extract_json(extracted)

        if parsed and "passed" in parsed:
            passed   = bool(parsed["passed"])
            feedback = parsed.get("feedback", "")
        else:
            console.print("[bold red]Eval JSON 解析失敗,不通過[/bold red]")
            passed   = False
            feedback = f"JSON 解析失败，原始：{raw[:200]}"

        if passed:
            self._accept(state)
            console.print("  [bold green]Eval 通過[/bold green]")
        else:
            state.eval_feedback    = feedback
            state.eval_retry_count += 1
            console.print(Panel(
                feedback,
                title=f"[bold red]Eval 不通过（已重试 {state.eval_retry_count} 次）[/bold red]",
                border_style="red",
            ))
            if state.eval_retry_count >= MAX_EVAL_RETRY:
                console.print("  [bold red]超出重試上限，使用最終輸出。[/bold red]")
                self._accept(state, force=True)

        state.log(
            Node.EVAL,
            "passed" if passed else f"retry#{state.eval_retry_count}",
            feedback[:60],
        )
        return state

    def _accept(self, state: GraphState, force: bool = False) -> None:
        step   = state.current_step
        output = state.current_output
        if force:
            output += "\n\n[警告] 超出重试上限，使用最后输出。"
        state.step_results.append({
            "step": step["step"],
            "task": step["description"],
            "result": output,
        })
        state.eval_feedback    = ""
        state.eval_retry_count = 0


class SummaryAgent(BaseAgent):
    def run(self, state: GraphState) -> GraphState:
        console.print(Rule("[bold yellow]SUMMARY[/bold yellow]"))

        steps_text = "\n".join(
            f"step{r['step']}：{r['task']}\nresult：{r['result']}"
            for r in state.step_results
        )

        system = """
你是一位负责重点整理的专家，根据所有步骤的执行结果，针对用户的原始任务给出最终回答。

流程：
1. 先针对「用户任务」与「各步骤执行结果」进行完整推理分析
2. 判断结果中是否有足够的实质相关资讯可以回答用户
3. 最后输出以下其中一种 result 区块：

若资讯足够：
```result
（一段话说明完成了什么流程）
（根据资讯回答用户问题，没有提供的资讯不可捏造）
```

范例：
```result
上网搜索了关于苹果手机的资讯并整理后存成了apple_smartphone_summary.txt档案
根据整理好的信息，苹果手机目前最新款式是...
```

若资讯不足或完全无关，特定格式（必须遵守格式）：
```result
[NEED_REPLAN]
```
"""
        output = self.llm.generate(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": (
                    f"用户原始任务：{state.user_task}\n\n"
                    f"各步骤执行结果：\n{steps_text}"
                )},
            ],
            label="總結Agent",
            max_tokens=8192,
        )
        clean_output  = _strip_think(output)
        match         = re.search(r"```result\s*(.*?)```", clean_output, re.DOTALL | re.IGNORECASE)
        result_content = match.group(1).strip() if match else ""

        if "[NEED_REPLAN]" in result_content:
            used_keywords = [
                step["input"][0]
                for step in state.workflow
                if step["skill_name"] == "wiki_skill" and step.get("input")
            ]
            for kw in used_keywords:
                if kw not in state.failed_keywords:
                    state.failed_keywords.append(kw)

            state.replan_count        += 1
            state.current_plan         = ""
            state.supervisor_feedback  = "[NEED_REPLAN] 搜索结果与任务无关，资讯不足，需换关键字重新规划。"
            state.workflow             = []
            state.step_results         = []
            state.current_output       = ""
            state.eval_feedback        = ""
            state.eval_retry_count     = 0
            state.current_step_index   = 0
            state._dispatched          = False

            console.print(Panel(
                f"資訊不足，replan（第 {state.replan_count} 次）\n已封鎖關鍵字：{state.failed_keywords}",
                title="[bold red]NEED_REPLAN[/bold red]",
                border_style="red",
            ))
        else:
            state.final_output = result_content or output

        state.log(Node.SUMMARY, "總結完成", output[:150])
        return state


def _strip_think(text: str) -> str:
    # 移除 <think>...</think> 区块，避免其中的内容干扰后续 regex
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict | None:
    text = _strip_think(text)
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*?\})",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _edge_supervise(state: GraphState) -> Node:
    return Node.DISPATCH if state.workflow else Node.PLAN


def _edge_dispatch(state: GraphState) -> Node:
    return Node.EXECUTE if state.current_step is not None else Node.SUMMARY


def _edge_eval(state: GraphState) -> Node:
    return Node.DISPATCH if not state.eval_feedback else Node.EXECUTE


def _edge_summary(state: GraphState) -> Node:
    MAX_REPLAN = 3
    if not state.final_output:
        if state.replan_count >= MAX_REPLAN:
            console.print(f"[bold red]Replan 超出 {MAX_REPLAN} 次上限，強制停止。[/bold red]")
            state.final_output = "（已嘗試多關鍵字但無法取得相關資訊）"
            return Node.END
        return Node.PLAN
    return Node.END


@dataclass
class GraphEdge:
    target:    Node | None                         = None
    condition: Callable[[GraphState], Node] | None = None


class AgentGraph:
    MAX_STEPS = 200

    def __init__(self, llm: LLMClient, skills_dir: str = "./skills", depth: int = 0):
        self._llm = llm
        self._skills_dir = skills_dir  # 存起來供 sub_workflow 用
        self._depth = depth
        skill_registry, skill_descriptions = load_skills(skills_dir)
        self._nodes = self._build_nodes(llm, skill_registry, skill_descriptions)
        self._graph = self._build_graph()

    def _build_nodes(self, llm, skill_registry, skill_descriptions) -> dict[Node, BaseAgent]:
        return {
            Node.PLAN:      PlanAgent(llm, skill_descriptions),
            Node.SUPERVISE: SuperviseAgent(llm, skill_descriptions),
            Node.DISPATCH:  DispatchAgent(llm),
            Node.EXECUTE:   ExecuteAgent(llm, skill_registry, skills_dir=self._skills_dir),
            Node.EVAL:      EvalAgent(llm),
            Node.SUMMARY:   SummaryAgent(llm),

        }

    def _build_graph(self) -> dict[Node, GraphEdge]:
        return {
            Node.PLAN:      GraphEdge(target=Node.SUPERVISE),
            Node.SUPERVISE: GraphEdge(condition=_edge_supervise),
            Node.DISPATCH:  GraphEdge(condition=_edge_dispatch),
            Node.EXECUTE:   GraphEdge(target=Node.EVAL),
            Node.EVAL:      GraphEdge(condition=_edge_eval),
            Node.SUMMARY:   GraphEdge(condition=_edge_summary),
            Node.END:       GraphEdge(target=None),
        }

    def run(self, user_task: str, sub_context: str = "") -> str:
        console.print(Panel(user_task, title=f"[bold]任務（depth={self._depth}）：[/bold]", border_style="white"))

        state        = GraphState(user_task=user_task,
                                  sub_context=sub_context,
                                  depth=self._depth
                                  )
        current_node = Node.PLAN

        for step_num in range(self.MAX_STEPS):
            if current_node == Node.END:
                break

            console.print(f"\n[bold dim]Step {step_num + 1}：[/bold dim][bold]{current_node.value}[/bold]")

            agent = self._nodes.get(current_node)
            if agent:
                state = agent.run(state)

            edge      = self._graph[current_node]
            next_node = edge.condition(state) if edge.condition else (edge.target or Node.END)

            console.print(f"  [dim]→ 下一節點：[bold]{next_node.value}[/bold][/dim]")
            current_node = next_node
        else:
            console.print(f"[bold red]超過步數上限 {self.MAX_STEPS}，強制停止。[/bold red]")

        self._print_log(state)
        self._print_step_results(state)
        console.print(Panel(
            state.final_output or "（無輸出）",
            title="[bold green]最終結果[/bold green]",
            border_style="green",
        ))
        return state.final_output

    @staticmethod
    def _print_log(state: GraphState) -> None:
        console.print(Rule("[bold white on green]  Graph 執行完成  [/bold white on green]"))
        table = Table(title="執行路徑紀錄", box=box.SIMPLE)
        table.add_column("步驟", style="dim",       width=5)
        table.add_column("節點", style="bold cyan",  width=12)
        table.add_column("動作", style="white")
        table.add_column("摘要", style="dim")
        for i, entry in enumerate(state.execution_log, 1):
            table.add_row(str(i), entry["node"], entry["action"], entry["detail"])
        console.print(table)

    @staticmethod
    def _print_step_results(state: GraphState) -> None:
        console.print(Rule("[bold cyan]Step Results[/bold cyan]"))
        console.print(Panel(
            json.dumps(state.step_results, ensure_ascii=False, indent=2),
            title="[cyan]step_results[/cyan]",
            border_style="cyan",
        ))


if __name__ == "__main__":
    llm   = LLMClient("/mnt/d/徐子家/實驗/qwen2.5-coder-14b-instruct-q8_0.gguf")
    graph = AgentGraph(llm, skills_dir="./skills")

    # graph.run("上wiki搜尋並整理一下 金色暗影 的資訊")
    # graph.run("世界上第一長河叫什麼名字")
    # graph.run("罗马帝国最终分裂是在西元几年")
    
    # graph.run("森亜るるか 是哪部动画的角色")
    
    # TODO:經由搜尋到的資訊繼續規劃下個搜尋關鍵字
    graph.run("金色暗影 的声优是谁?她还有为哪些作品配音?")