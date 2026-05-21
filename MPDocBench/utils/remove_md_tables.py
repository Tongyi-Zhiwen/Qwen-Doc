# -*- coding: utf-8 -*-
"""
工具类：从 pdfs 目录下的 Markdown 文件中移除所有表格，其他内容保持不变。
"""

import re
from pathlib import Path


def _is_table_row(line: str) -> bool:
    """判断一行是否为 Markdown 表格行（含表头、分隔行、数据行）。"""
    s = line.strip()
    if not s:
        return False
    # 表格行：以 | 开头、以 | 结尾，且至少包含两个 |（即至少一列）
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def remove_tables_from_content(content: str) -> str:
    """
    从 Markdown 文本中移除所有表格行，其余内容（含空行、标题、段落等）不变。

    Args:
        content: 原始 Markdown 文本

    Returns:
        去掉表格后的 Markdown 文本
    """
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    in_table = False

    for line in lines:
        if _is_table_row(line):
            in_table = True
            continue
        # 非表格行
        if in_table:
            # 表格刚结束，可选：保留一个空行避免与前后文贴在一起
            in_table = False
        result.append(line)

    return "".join(result)


def remove_tables_from_file(file_path: str | Path) -> None:
    """
    原地修改单个 Markdown 文件，移除其中所有表格。

    Args:
        file_path: .md 文件路径
    """
    path = Path(file_path)
    if not path.is_file() or path.suffix.lower() != ".md":
        return
    content = path.read_text(encoding="utf-8")
    new_content = remove_tables_from_content(content)
    if content != new_content:
        path.write_text(new_content, encoding="utf-8")


def remove_tables_from_pdfs_dir(pdfs_dir: str | Path = "pdfs") -> list[Path]:
    """
    遍历 pdfs 目录（及其子目录）下所有 .md 文件，移除其中的表格并写回。

    Args:
        pdfs_dir: 存放 md 的目录，默认为项目下的 "pdfs"

    Returns:
        被修改过的 .md 文件路径列表
    """
    root = Path(pdfs_dir)
    if not root.is_dir():
        return []
    modified: list[Path] = []
    for md_path in root.rglob("*.md"):
        content = md_path.read_text(encoding="utf-8")
        new_content = remove_tables_from_content(content)
        if content != new_content:
            md_path.write_text(new_content, encoding="utf-8")
            modified.append(md_path)
    return modified


if __name__ == "__main__":
    import sys

    pdfs = Path(__file__).resolve().parent.parent / "pdfs"
    if len(sys.argv) > 1:
        pdfs = Path(sys.argv[1])
    changed = remove_tables_from_pdfs_dir(pdfs)
    print(f"已处理目录: {pdfs}")
    print(f"已移除表格的文件数: {len(changed)}")
    for p in changed[:20]:
        print(f"  - {p.relative_to(pdfs) if p.is_relative_to(pdfs) else p}")
    if len(changed) > 20:
        print(f"  ... 及其他 {len(changed) - 20} 个文件")
