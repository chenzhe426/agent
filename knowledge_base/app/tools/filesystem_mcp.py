"""
filesystem_mcp.py - 文件系统 MCP 工具

提供本地文件系统操作的 MCP 工具：
- read_file: 读取文件内容
- list_directory: 列出目录内容
- write_file: 写入文件内容
- create_directory: 创建目录
- delete_file: 删除文件
- file_exists: 检查文件是否存在
- get_file_info: 获取文件信息

使用 FastMCP 的 Path 参数类型，自动处理路径验证。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from fastmcp import FastMCP
except ImportError:
    # Fallback - will be checked in mcp_server.py
    FastMCP = None

from pydantic import Field


def register_filesystem_tools(mcp: FastMCP) -> None:
    """
    注册所有文件系统 MCP 工具到指定的 FastMCP 实例。

    Args:
        mcp: FastMCP 实例
    """

    @mcp.tool(name="filesystem_read_file")
    def mcp_read_file(
        path: Path = Field(..., description="Path to the file to read."),
        encoding: str = Field("utf-8", description="File encoding (default: utf-8)."),
        max_lines: Optional[int] = Field(None, description="Maximum number of lines to read."),
    ) -> str:
        """
        读取文件内容。

        适用于读取文本文件、配置文件、代码文件等。
        大文件建议指定 max_lines 参数。
        """
        try:
            if not path.exists():
                return f"Error: File not found: {path}"

            if not path.is_file():
                return f"Error: Not a file: {path}"

            content = path.read_text(encoding=encoding)

            if max_lines is not None:
                lines = content.split("\n")
                content = "\n".join(lines[:max_lines])
                if len(lines) > max_lines:
                    content += f"\n... (truncated, {len(lines) - max_lines} more lines)"

            return content

        except UnicodeDecodeError:
            return f"Error: Unable to decode file with {encoding} encoding. Try reading as binary."
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @mcp.tool(name="filesystem_write_file")
    def mcp_write_file(
        path: Path = Field(..., description="Path to the file to write."),
        content: str = Field(..., description="Content to write to the file."),
        encoding: str = Field("utf-8", description="File encoding (default: utf-8)."),
        create_parent_dirs: bool = Field(True, description="Create parent directories if they don't exist."),
    ) -> str:
        """
        写入内容到文件。

        如果文件存在，会覆盖原有内容。
        默认会创建父目录。
        """
        try:
            if create_parent_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding=encoding)
            return f"Success: Wrote {len(content)} bytes to {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @mcp.tool(name="filesystem_list_directory")
    def mcp_list_directory(
        path: Path = Field(..., description="Path to the directory to list."),
        include_hidden: bool = Field(False, description="Include hidden files (starting with .)."),
    ) -> str:
        """
        列出目录内容。

        返回格式：
        [type] name (size/info)
        d = directory, f = file
        """
        try:
            if not path.exists():
                return f"Error: Directory not found: {path}"

            if not path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(path.iterdir()):
                if not include_hidden and item.name.startswith("."):
                    continue

                if item.is_dir():
                    # 尝试计算目录中的文件数
                    try:
                        subcount = len(list(item.iterdir()))
                        info = f"{subcount} items"
                    except PermissionError:
                        info = "permission denied"
                    items.append(f"[d] {item.name} ({info})")
                else:
                    size = item.stat().st_size
                    size_str = _format_size(size)
                    items.append(f"[f] {item.name} ({size_str})")

            if not items:
                return f"Directory is empty: {path}"

            return "\n".join(items)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @mcp.tool(name="filesystem_create_directory")
    def mcp_create_directory(
        path: Path = Field(..., description="Path to the directory to create."),
        parents: bool = Field(True, description="Create parent directories if they don't exist."),
    ) -> str:
        """
        创建目录。

        默认会创建父目录（类似 mkdir -p）。
        """
        try:
            path.mkdir(parents=parents, exist_ok=True)
            return f"Success: Created directory {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    @mcp.tool(name="filesystem_delete")
    def mcp_delete(
        path: Path = Field(..., description="Path to the file or directory to delete."),
        recursive: bool = Field(False, description="Delete directories recursively."),
    ) -> str:
        """
        删除文件或目录。

        默认不递归删除目录。如需删除目录及其内容，使用 recursive=True。
        """
        try:
            if not path.exists():
                return f"Error: Path does not exist: {path}"

            if path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                    return f"Success: Deleted directory {path}"
                else:
                    path.rmdir()
                    return f"Success: Deleted empty directory {path}"
            else:
                path.unlink()
                return f"Success: Deleted file {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except OSError as e:
            if "directory not empty" in str(e).lower():
                return f"Error: Directory not empty. Use recursive=True to delete."
            return f"Error deleting path: {str(e)}"
        except Exception as e:
            return f"Error deleting path: {str(e)}"

    @mcp.tool(name="filesystem_exists")
    def mcp_exists(
        path: Path = Field(..., description="Path to check."),
    ) -> str:
        """
        检查文件或目录是否存在。

        返回简单的存在性信息。
        """
        if path.exists():
            if path.is_dir():
                return f"Exists: {path} (directory)"
            else:
                size = path.stat().st_size
                return f"Exists: {path} (file, {_format_size(size)})"
        else:
            return f"Not found: {path}"

    @mcp.tool(name="filesystem_get_info")
    def mcp_get_info(
        path: Path = Field(..., description="Path to get information about."),
    ) -> str:
        """
        获取文件或目录的详细信息。

        包括大小、创建时间、修改时间、权限等。
        """
        try:
            if not path.exists():
                return f"Error: Path does not exist: {path}"

            stat = path.stat()

            info_lines = [
                f"Path: {path}",
                f"Type: {'directory' if path.is_dir() else 'file'}",
                f"Size: {_format_size(stat.st_size)}",
                f"Created: {datetime.fromtimestamp(stat.st_ctime).isoformat()}",
                f"Modified: {datetime.fromtimestamp(stat.st_mtime).isoformat()}",
                f"Accessed: {datetime.fromtimestamp(stat.st_atime).isoformat()}",
            ]

            # 尝试获取权限
            try:
                mode = stat.st_mode
                perms = []
                for who in "USR", "GRP", "OTH":
                    for what in "R", "W", "X":
                        if mode & getattr(os, f"S_I{what}{who}"):
                            perms.append(f"{who[0]}{what}")
                        else:
                            perms.append("-")
                info_lines.append(f"Permissions: {''.join(perms)}")
            except Exception:
                pass

            return "\n".join(info_lines)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error getting info: {str(e)}"

    @mcp.tool(name="filesystem_copy")
    def mcp_copy(
        source: Path = Field(..., description="Source path."),
        destination: Path = Field(..., description="Destination path."),
        overwrite: bool = Field(False, description="Overwrite if destination exists."),
    ) -> str:
        """
        复制文件或目录。

        默认不覆盖已存在的目标。
        """
        try:
            if not source.exists():
                return f"Error: Source does not exist: {source}"

            if destination.exists() and not overwrite:
                return f"Error: Destination exists: {destination}. Use overwrite=True to overwrite."

            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
                return f"Success: Copied directory {source} to {destination}"
            else:
                shutil.copy2(source, destination)
                return f"Success: Copied file {source} to {destination}"

        except PermissionError:
            return f"Error: Permission denied"
        except Exception as e:
            return f"Error copying: {str(e)}"

    @mcp.tool(name="filesystem_move")
    def mcp_move(
        source: Path = Field(..., description="Source path."),
        destination: Path = Field(..., description="Destination path."),
        overwrite: bool = Field(False, description="Overwrite if destination exists."),
    ) -> str:
        """
        移动/重命名文件或目录。
        """
        try:
            if not source.exists():
                return f"Error: Source does not exist: {source}"

            if destination.exists() and not overwrite:
                return f"Error: Destination exists: {destination}. Use overwrite=True to overwrite."

            if destination.exists() and overwrite:
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()

            shutil.move(str(source), str(destination))
            return f"Success: Moved {source} to {destination}"

        except PermissionError:
            return f"Error: Permission denied"
        except Exception as e:
            return f"Error moving: {str(e)}"

    @mcp.tool(name="filesystem_search")
    def mcp_search(
        directory: Path = Field(..., description="Directory to search in."),
        pattern: str = Field(..., description="Filename pattern to match (supports * and ?)."),
        recursive: bool = Field(True, description="Search recursively in subdirectories."),
        max_results: int = Field(50, ge=1, le=500, description="Maximum number of results to return."),
    ) -> str:
        """
        在目录中搜索匹配指定模式的文件。

        支持通配符：
        - * 匹配任意字符
        - ? 匹配单个字符
        """
        try:
            if not directory.exists():
                return f"Error: Directory does not exist: {directory}"

            if not directory.is_dir():
                return f"Error: Not a directory: {directory}"

            import fnmatch

            results = []
            if recursive:
                for root, dirs, files in os.walk(directory):
                    # 跳过隐藏目录
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for name in files:
                        if fnmatch.fnmatch(name, pattern):
                            full_path = Path(root) / name
                            try:
                                size = full_path.stat().st_size
                                results.append(f"[f] {full_path} ({_format_size(size)})")
                            except PermissionError:
                                results.append(f"[f] {full_path} (permission denied)")
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            else:
                for item in directory.iterdir():
                    if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                        try:
                            size = item.stat().st_size
                            results.append(f"[f] {item} ({_format_size(size)})")
                        except PermissionError:
                            results.append(f"[f] {item} (permission denied)")
                        if len(results) >= max_results:
                            break

            if not results:
                return f"No files matching '{pattern}' found in {directory}"

            if len(results) >= max_results:
                return f"Found {len(results)}+ matching files:\n" + "\n".join(results)
            return f"Found {len(results)} matching files:\n" + "\n".join(results)

        except PermissionError:
            return f"Error: Permission denied: {directory}"
        except Exception as e:
            return f"Error searching: {str(e)}"


def _format_size(size: int) -> str:
    """格式化文件大小为可读字符串"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
