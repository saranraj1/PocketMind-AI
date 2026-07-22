import os
import re
from pathlib import Path
from pocketmind.actions.schema import (
    PocketMindAction, SearchTextAction, ListFilesAction,
    ReadFileSectionAction, FindTodosAction
)

class ActionExecutor:
    def __init__(self, workspace_root: str | Path):
        self.workspace_root = os.path.realpath(workspace_root)

    def _validate_and_resolve_path(self, relative_path: str) -> str:
        # 1) Combine and compute real canonical absolute path (resolves symlinks/..)
        joined_path = os.path.join(self.workspace_root, relative_path)
        abs_path = os.path.realpath(joined_path)
        
        # 2) Assert path containment: abs_path must reside inside workspace_root
        try:
            common = os.path.commonpath([self.workspace_root, abs_path])
        except ValueError:
            # Raised if on different drives (Windows)
            raise PermissionError(f"Access Denied: Path escapes workspace boundary ({relative_path})")
            
        if common != self.workspace_root:
            raise PermissionError(f"Access Denied: Path escapes workspace boundary ({relative_path})")
            
        return abs_path

    def execute(self, action_wrapper: PocketMindAction) -> str:
        action_data = action_wrapper.root
        action_type = action_data.action
        
        try:
            if isinstance(action_data, SearchTextAction):
                return self._execute_search_text(action_data)
            elif isinstance(action_data, ListFilesAction):
                return self._execute_list_files(action_data)
            elif isinstance(action_data, ReadFileSectionAction):
                return self._execute_read_file_section(action_data)
            elif isinstance(action_data, FindTodosAction):
                return self._execute_find_todos(action_data)
            else:
                # Meta actions (summarize_results, ask_clarification, refuse_action) are no-ops for executor
                return f"No execution required for action: {action_type}"
        except Exception as e:
            return f"Execution Error: {str(e)}"

    def _execute_list_files(self, action: ListFilesAction) -> str:
        abs_dir = self._validate_and_resolve_path(action.arguments.path)
        if not os.path.isdir(abs_dir):
            return f"Error: Path {action.arguments.path} is not a directory."
            
        try:
            entries = os.listdir(abs_dir)
            files = []
            dirs = []
            for entry in sorted(entries):
                full_p = os.path.join(abs_dir, entry)
                if os.path.isdir(full_p):
                    dirs.append(entry + "/")
                else:
                    files.append(entry)
            result = f"Contents of directory '{action.arguments.path}':\n"
            result += "Directories:\n" + "\n".join(dirs) if dirs else "No directories found.\n"
            result += "\nFiles:\n" + "\n".join(files) if files else "No files found.\n"
            return result
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def _execute_read_file_section(self, action: ReadFileSectionAction) -> str:
        abs_file = self._validate_and_resolve_path(action.arguments.path)
        if not os.path.isfile(abs_file):
            return f"Error: Path {action.arguments.path} is not a file."
            
        start = action.arguments.start_line
        end = action.arguments.end_line
        if start > end:
            return f"Error: Invalid line range: start_line ({start}) > end_line ({end})."
            
        # Limit total lines read to prevent VRAM/RAM flooding
        line_limit = 500
        if (end - start + 1) > line_limit:
            end = start + line_limit - 1
            truncation_notice = f"\n[Notice: Truncated to {line_limit} lines limit]"
        else:
            truncation_notice = ""
            
        try:
            # Check if file is binary by reading a block
            with open(abs_file, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return f"Error: File '{action.arguments.path}' is binary and cannot be read."
                    
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            # Slice lines (1-based index)
            sliced = lines[start - 1 : end]
            numbered = [f"{start + idx}: {line}" for idx, line in enumerate(sliced)]
            return f"File '{action.arguments.path}' (lines {start}-{end}):\n" + "".join(numbered) + truncation_notice
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _execute_search_text(self, action: SearchTextAction) -> str:
        abs_dir = self._validate_and_resolve_path(action.arguments.path)
        if not os.path.isdir(abs_dir):
            return f"Error: Path {action.arguments.path} is not a directory."
            
        try:
            pattern = re.compile(action.arguments.pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {str(e)}"
            
        extensions = set(action.arguments.extensions)
        limit = action.arguments.limit
        matches = []
        
        # Recursively search files
        for root, _, files in os.walk(abs_dir):
            for file in files:
                suffix = Path(file).suffix
                if suffix not in extensions:
                    continue
                    
                full_p = os.path.join(root, file)
                # Ensure path is valid workspace containment
                try:
                    abs_file = self._validate_and_resolve_path(os.path.relpath(full_p, self.workspace_root))
                except PermissionError:
                    continue
                    
                try:
                    # Ignore binary files
                    with open(abs_file, "rb") as f:
                        if b"\x00" in f.read(1024):
                            continue
                            
                    with open(abs_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f):
                            if pattern.search(line):
                                rel_file = os.path.relpath(abs_file, self.workspace_root)
                                matches.append(f"{rel_file}:{line_idx + 1}: {line.strip()}")
                                if len(matches) >= limit:
                                    break
                except Exception:
                    continue
                    
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break
                
        result = f"Search matches for '{action.arguments.pattern}':\n"
        if matches:
            result += "\n".join(matches)
        else:
            result += "No matches found."
        return result

    def _execute_find_todos(self, action: FindTodosAction) -> str:
        # Re-use search_text with pre-configured pattern
        from pocketmind.actions.schema import SearchTextAction, SearchTextArguments
        search_action = PocketMindAction(
            SearchTextAction(
                arguments=SearchTextArguments(
                    pattern=r"TODO|FIXME",
                    path=action.arguments.path,
                    limit=50
                )
            )
        )
        return self.execute(search_action)
