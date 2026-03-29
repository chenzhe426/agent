#!/usr/bin/env python3
"""Export chat data to ShareGPT format for SFT training."""

import json
from pathlib import Path

from app.db.connection import get_cursor
from app.db.repositories.chat_repository import get_chat_messages


def export_to_sharegpt(output_path: str = "sft_data.jsonl"):
    """Export all chat sessions to ShareGPT format."""

    with get_cursor() as (_, cursor):
        # Get all session IDs
        cursor.execute("""
            SELECT session_id
            FROM chat_sessions
            ORDER BY created_at DESC
        """)
        sessions = cursor.fetchall() or []

    sharegpt_data = []

    for session_row in sessions:
        session_id = session_row["session_id"]
        messages = get_chat_messages(session_id, limit=None)

        if not messages:
            continue

        # Skip sessions with only one message (need at least a user and assistant)
        if len(messages) < 2:
            continue

        # Convert to ShareGPT conversations format (LlamaFactory standard)
        # ShareGPT format: {"conversations": [{"from": "human"|"gpt"|"system", "value": "..."}]}
        conversations = []
        for msg in messages:
            role = msg["role"]
            # Map role names to ShareGPT "from" format
            if role == "user":
                from_role = "human"
            elif role == "assistant":
                from_role = "gpt"
            elif role == "system":
                from_role = "system"
            else:
                # Skip unknown roles
                continue

            conversations.append({
                "from": from_role,
                "value": msg["message"]
            })

        # Only keep valid conversations (alternating user/assistant)
        if len(conversations) >= 2:
            sharegpt_data.append({"conversations": conversations})

    # Write to JSONL format
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sharegpt_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported {len(sharegpt_data)} conversations to {output_path}")
    return sharegpt_data


if __name__ == "__main__":
    export_to_sharegpt()