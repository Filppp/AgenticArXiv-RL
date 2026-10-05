"""Tests for schema-safe deterministic arguments after an external route."""

from benchmark.tasks_expanded import EXPANDED_TASKS
from routing.arguments import resolve_routed_arguments


def test_extracts_explicit_keyword_query_window_and_limit():
    result = resolve_routed_arguments(
        tool_name="search_arxiv_papers",
        task=(
            "按关键词检索 arXiv：all:agentic reinforcement learning，"
            "最近 30 天，最多返回 5 篇论文"
        ),
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {
        "query": "all:agentic reinforcement learning",
        "days": 30,
        "max_results": 5,
    }


def test_extracts_direct_arxiv_id_without_requiring_candidate_membership():
    result = resolve_routed_arguments(
        tool_name="download_arxiv_pdf",
        task="下载 2608.14528v1 这篇论文",
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {"ref": "2608.14528v1"}


def test_extracts_translation_reference_and_service():
    result = resolve_routed_arguments(
        tool_name="translate_arxiv_pdf",
        task="用 google 服务翻译第1篇论文",
        history="",
    )
    assert result.args == {"ref": 1, "service": "google"}


def test_active_reference_is_explicit_null():
    result = resolve_routed_arguments(
        tool_name="get_paper_cache_status",
        task="查一下我刚下载的那篇论文的缓存状态",
        history="",
    )
    assert result.args == {"ref": None}


def test_non_positive_reference_is_blocked_before_execution():
    result = resolve_routed_arguments(
        tool_name="download_arxiv_pdf",
        task="下载第0篇论文",
        history="",
    )
    assert result.status == "blocked"
    assert result.args == {}
    assert result.reason == "non_positive_reference"


def test_multi_reference_advances_then_reports_completion():
    task = "第1篇和第2篇，分别查一下缓存状态"
    first = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=""
    )
    assert first.args == {"ref": 1}
    history_one = (
        'Thought: x\nAction: {"name":"get_paper_cache_status",'
        '"args":{"ref":1}}\nObservation: ok'
    )
    second = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=history_one
    )
    assert second.args == {"ref": 2}
    history_two = history_one + (
        '\n\nThought: x\nAction: {"name":"get_paper_cache_status",'
        '"args":{"ref":2}}\nObservation: ok'
    )
    complete = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=history_two
    )
    assert complete.status == "complete"


# ---------------------------------------------------------------------------
# 读译文：ref 复用同一套指代解析，page 是本工具独有的显式参数
# ---------------------------------------------------------------------------


def test_translated_reader_extracts_reference_and_explicit_page():
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="打开第1篇论文的中文译文，读第2页",
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {"ref": 1, "page": 2}


def test_translated_reader_leaves_page_to_the_tool_default():
    """不提页数时不写 page —— 省略即第 1 页，不是猜出来的。"""
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读一下第1篇论文的中文译文开头（标题与摘要）",
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {"ref": 1}


def test_translated_reader_accepts_arxiv_id_and_page():
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读一下 2608.14528v1 的中文译文第3页",
        history="",
    )
    assert result.args == {"ref": "2608.14528v1", "page": 3}


def test_translated_reader_accepts_active_reference_and_page():
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读一下我刚下载的那篇论文的译文第2页",
        history="",
    )
    assert result.args == {"ref": None, "page": 2}


def test_translated_reader_defers_when_several_pages_are_named():
    """「第2页和第3页」是列举：不能挑第 2 页当确定参数。"""
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读第1篇译文的第2页和第3页",
        history="",
    )
    assert result.status == "defer"
    assert result.reason == "translated_page_ambiguous"
    assert result.args == {}


def test_translated_reader_defers_on_a_page_range_instead_of_defaulting():
    """「第2-3页」不含独立的「第N页」；此时退回默认第 1 页是错的参数。"""
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读第1篇译文的第2-3页",
        history="",
    )
    assert result.status == "defer"
    assert result.reason == "translated_page_not_single"
    assert result.args == {}


def test_translated_reader_defers_on_non_positive_page():
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="读第1篇译文的第0页",
        history="",
    )
    assert result.status == "defer"
    assert result.reason == "translated_page_not_positive"


def test_translated_reader_keeps_ambiguous_active_phrasing_with_the_policy():
    """「刚才翻译好的那篇」不在显式活跃指代词表里，仍交给策略模型。

    词表是刻意收窄的（见 routing/arguments.py 的 active_markers），扩词会
    影响所有论文工具；这里把它钉住，避免被当成实现缺陷悄悄改掉。
    """
    result = resolve_routed_arguments(
        tool_name="get_translated_content",
        task="把刚才翻译好的那篇论文的译文第3页读给我",
        history="",
    )
    assert result.status == "defer"
    assert result.reason == "paper_reference_not_explicit"


def test_translated_reading_tasks_never_resolve_to_wrong_arguments():
    """扩集里的读译文任务：解析器要么给出标准答案，要么交回策略，绝不给错。"""
    tasks = [t for t in EXPANDED_TASKS if t["category"] == "translation_reading"]
    assert tasks
    deterministic = []
    for task in tasks:
        expected = task["expected_tool_args"][0]
        result = resolve_routed_arguments(
            tool_name="get_translated_content", task=task["task"], history=""
        )
        if result.status != "resolved":
            continue
        deterministic.append(task["id"])
        assert result.args.get("ref") == expected.get("ref"), task["id"]
        # 省略 page 与显式传 1 等价，与 metrics._match_arg_value 同口径。
        assert result.args.get("page", 1) == expected.get("page", 1), task["id"]
    # 只剩「把刚才翻译好的那篇论文的译文第3页读给我」按设计交回策略模型
    # （该措辞不在活跃指代词表里），其余都该走确定性参数。
    assert len(deterministic) >= 4
