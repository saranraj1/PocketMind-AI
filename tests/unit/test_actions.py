import os
import tempfile
import pytest
from pydantic import ValidationError
from pocketmind.actions.schema import PocketMindAction, SearchTextAction, SearchTextArguments
from pocketmind.actions.executor import ActionExecutor
from pocketmind.actions.constrained_decode import JSONGrammarParser

def test_pydantic_action_schemas():
    # Valid Action
    valid_data = {
        "version": "1",
        "action": "search_text",
        "arguments": {
            "pattern": "TODO|FIXME",
            "path": "src",
            "limit": 10
        }
    }
    action = PocketMindAction.model_validate(valid_data)
    assert action.root.action == "search_text"
    assert action.root.arguments.pattern == "TODO|FIXME"
    
    # Invalid Action (Unknown field)
    invalid_data = {
        "version": "1",
        "action": "search_text",
        "arguments": {
            "pattern": "TODO",
            "invalid_extra_field": True
        }
    }
    with pytest.raises(ValidationError):
        PocketMindAction.model_validate(invalid_data)
        
    # Invalid Argument Constraint (out of bounds limit)
    invalid_limit_data = {
        "version": "1",
        "action": "search_text",
        "arguments": {
            "pattern": "TODO",
            "limit": 1000  # ge=1, le=100
        }
    }
    with pytest.raises(ValidationError):
        PocketMindAction.model_validate(invalid_limit_data)

def test_executor_security_containment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock file inside workspace root
        inside_file = os.path.join(tmp_dir, "test.py")
        with open(inside_file, "w") as f:
            f.write("print('hello')\n# TODO: fix this")
            
        executor = ActionExecutor(workspace_root=tmp_dir)
        
        # 1) Accessing file inside should work
        from pocketmind.actions.schema import ReadFileSectionAction, ReadFileSectionArguments
        action_read = PocketMindAction(
            ReadFileSectionAction(
                arguments=ReadFileSectionArguments(path="test.py", start_line=1, end_line=2)
            )
        )
        res = executor.execute(action_read)
        assert "print('hello')" in res
        
        # 2) Accessing file outside (directory traversal) should raise PermissionError
        action_escape = PocketMindAction(
            ReadFileSectionAction(
                arguments=ReadFileSectionArguments(path="../escaped.txt", start_line=1, end_line=5)
            )
        )
        res_escape = executor.execute(action_escape)
        assert "Access Denied" in res_escape or "PermissionError" in res_escape

def test_json_grammar_parser():
    parser = JSONGrammarParser()
    
    # 1) Expect brace first
    assert parser.consume("{")
    assert not parser.consume("]") # Bracket mismatch fails (expected '}' to close brace)
    
    # 2) Expect key starting with double quote
    assert parser.consume('"')
    assert parser.in_string
    
    # Characters inside string should be allowed
    assert parser.consume("a")
    assert parser.consume("b")
    assert parser.consume('"') # Close string
    assert not parser.in_string
    
    # Expect colon after key
    assert parser.consume(":")
    
    # Expect value (e.g. number)
    assert parser.consume("1")
    assert parser.consume("2")
    
    # Expect brace close
    assert parser.consume("}")
    
    # Stack should be empty now
    assert len(parser.stack) == 0
