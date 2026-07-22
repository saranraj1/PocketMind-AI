from typing import Literal, Union, List, Annotated
from pydantic import BaseModel, Field, RootModel

class StrictBaseModel(BaseModel):
    model_config = {"extra": "forbid"}

# 1) SearchText Arguments and Schema
class SearchTextArguments(StrictBaseModel):
    pattern: str = Field(..., max_length=128, description="Regex or text search pattern.")
    path: str = Field(default=".", description="Relative path within workspace.")
    extensions: List[str] = Field(
        default_factory=lambda: [".py", ".ts", ".tsx", ".md", ".json", ".yaml", ".c", ".cpp", ".h"],
        description="List of file extensions to include in search."
    )
    limit: int = Field(default=50, ge=1, le=100, description="Max matches to return.")

class SearchTextAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["search_text"] = "search_text"
    arguments: SearchTextArguments

# 2) ListFiles Arguments and Schema
class ListFilesArguments(StrictBaseModel):
    path: str = Field(default=".", description="Directory path to list.")

class ListFilesAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["list_files"] = "list_files"
    arguments: ListFilesArguments

# 3) ReadFileSection Arguments and Schema
class ReadFileSectionArguments(StrictBaseModel):
    path: str = Field(..., description="File path to read.")
    start_line: int = Field(..., ge=1, description="Start line index (1-based).")
    end_line: int = Field(..., ge=1, description="End line index (inclusive).")

class ReadFileSectionAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["read_file_section"] = "read_file_section"
    arguments: ReadFileSectionArguments

# 4) FindTodos Arguments and Schema
class FindTodosArguments(StrictBaseModel):
    path: str = Field(default=".", description="Directory path to scan.")

class FindTodosAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["find_todos"] = "find_todos"
    arguments: FindTodosArguments

# 5) SummarizeResults Arguments and Schema
class SummarizeResultsArguments(StrictBaseModel):
    summary: str = Field(..., max_length=1000, description="Final response summary.")

class SummarizeResultsAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["summarize_results"] = "summarize_results"
    arguments: SummarizeResultsArguments

# 6) AskClarification Arguments and Schema
class AskClarificationArguments(StrictBaseModel):
    question: str = Field(..., max_length=500, description="Question text to ask.")

class AskClarificationAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["ask_clarification"] = "ask_clarification"
    arguments: AskClarificationArguments

# 7) RefuseAction Arguments and Schema
class RefuseActionArguments(StrictBaseModel):
    reason: str = Field(..., max_length=500, description="Reason for policy refusal.")

class RefuseActionAction(StrictBaseModel):
    version: Literal["1"] = "1"
    action: Literal["refuse_action"] = "refuse_action"
    arguments: RefuseActionArguments

# Discriminated Union representing all allowed actions
PocketMindActionUnion = Annotated[
    Union[
        SearchTextAction,
        ListFilesAction,
        ReadFileSectionAction,
        FindTodosAction,
        SummarizeResultsAction,
        AskClarificationAction,
        RefuseActionAction
    ],
    Field(discriminator="action")
]

class PocketMindAction(RootModel[PocketMindActionUnion]):
    pass
