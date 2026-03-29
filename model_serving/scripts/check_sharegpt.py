#!/usr/bin/env python3
"""
检查评测数据格式是否符合 ShareGPT 标准

ShareGPT 标准格式:
  - conversations: [{"from": "human"|"gpt", "value": "..."}]

Usage:
    python scripts/check_sharegpt.py evals/datasets/kb_eval_sft.jsonl
"""
import json
import sys
from pathlib import Path


def check_sharegpt_format(file_path: str) -> dict:
    """检查文件是否符合 ShareGPT 格式"""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    results = {
        "file": str(file_path),
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "format_issues": [],
        "samples": [],
    }

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            results["total_lines"] += 1

            try:
                data = json.loads(line)

                # 检查必要字段
                has_conversations = "conversations" in data
                has_messages = "messages" in data

                if not has_conversations and not has_messages:
                    results["invalid_lines"] += 1
                    results["format_issues"].append(f"Line {i}: Missing 'conversations' or 'messages' field")
                    continue

                # 检查 conversations 格式 (LlamaFactory/标准ShareGPT)
                if has_conversations:
                    convs = data["conversations"]
                    if not isinstance(convs, list):
                        results["invalid_lines"] += 1
                        results["format_issues"].append(f"Line {i}: 'conversations' is not a list")
                        continue

                    valid_conv = True
                    for j, msg in enumerate(convs):
                        if "from" not in msg or "value" not in msg:
                            results["format_issues"].append(
                                f"Line {i}: conversation[{j}] missing 'from' or 'value'"
                            )
                            valid_conv = False
                            break

                    if valid_conv:
                        results["valid_lines"] += 1
                        if len(results["samples"]) < 3:
                            results["samples"].append({
                                "line": i,
                                "format": "conversations",
                                "messages_count": len(convs),
                                "preview": convs[0].get("value", "")[:100],
                            })

                # 检查 messages 格式 (HuggingFace/OpenAI)
                elif has_messages:
                    msgs = data["messages"]
                    if not isinstance(msgs, list):
                        results["invalid_lines"] += 1
                        results["format_issues"].append(f"Line {i}: 'messages' is not a list")
                        continue

                    valid_msg = True
                    for j, msg in enumerate(msgs):
                        if "role" not in msg or "content" not in msg:
                            results["format_issues"].append(
                                f"Line {i}: message[{j}] missing 'role' or 'content'"
                            )
                            valid_msg = False
                            break

                    if valid_msg:
                        results["valid_lines"] += 1
                        if len(results["samples"]) < 3:
                            results["samples"].append({
                                "line": i,
                                "format": "messages",
                                "messages_count": len(msgs),
                                "preview": msgs[0].get("content", "")[:100],
                            })

            except json.JSONDecodeError as e:
                results["invalid_lines"] += 1
                results["format_issues"].append(f"Line {i}: JSON parse error - {e}")

    return results


def print_report(results: dict):
    """打印格式检查报告"""
    print("=" * 70)
    print(f"ShareGPT Format Check Report")
    print("=" * 70)

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nFile: {results['file']}")
    print(f"Total lines: {results['total_lines']}")
    print(f"Valid lines: {results['valid_lines']}")
    print(f"Invalid lines: {results['invalid_lines']}")

    if results["valid_lines"] == results["total_lines"]:
        print("\n✓ All lines are valid ShareGPT format")
    else:
        print(f"\n✗ {results['invalid_lines']} lines have format issues")

    if results["format_issues"]:
        print(f"\nIssues found ({len(results['format_issues'])}):")
        for issue in results["format_issues"][:10]:
            print(f"  - {issue}")
        if len(results["format_issues"]) > 10:
            print(f"  ... and {len(results['format_issues']) - 10} more")

    if results["samples"]:
        print(f"\nSample records:")
        for sample in results["samples"]:
            print(f"\n  Line {sample['line']} ({sample['format']} format, {sample['messages_count']} messages):")
            print(f"    Preview: {sample['preview'][:80]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_sharegpt.py <jsonl_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    results = check_sharegpt_format(file_path)
    print_report(results)

    # Exit with error code if issues found
    if results.get("invalid_lines", 0) > 0:
        sys.exit(1)
