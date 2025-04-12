from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E902

import pytest

from main import (
    display_results,
    parse_arguments,
    process_query,
    run_with_args,
    stream_dataframe_mode,
    stream_sql_mode,
)


def test_parse_arguments():
    with patch("sys.argv", ["script.py", "--prompt", "test", "--mode", "sql", "--db", "sqlite:///:memory:"]):
        args = parse_arguments()
    assert args.prompt == "test"
    assert args.mode == "sql"
    assert args.db == "sqlite:///:memory:"


@pytest.mark.asyncio
async def test_run_with_args_sql_mode():
    args = MagicMock()
    args.prompt = "test"
    args.mode = "sql"
    args.db = "sqlite:///:memory:"
    args.stream = False
    args.token_limit = 4000
    args.request_limit = 10
    with patch("app.agent_orchestrator.run_agent_orchestrator", new=AsyncMock(return_value={"success": True})):
        result = await run_with_args(args)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_run_with_args_no_prompt():
    args = MagicMock()
    args.prompt = None
    args.mode = "sql"
    args.db = "sqlite:///:memory:"
    args.stream = False
    args.token_limit = 4000
    args.request_limit = 10
    with (
        patch("builtins.input", return_value="test"),
        patch("app.agent_orchestrator.run_agent_orchestrator", new=AsyncMock(return_value={"success": True})),
        patch("sqlalchemy.create_engine") as mock_engine,
    ):
        mock_engine.return_value.connect.return_value = MagicMock()
        result = await run_with_args(args)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_stream_sql_mode(usage_limits, mock_db_connection):  # noqa: PT019
    args = MagicMock()
    args.prompt = "test"
    args.db = "sqlite:///:memory:"
    with (
        patch("sqlalchemy.create_engine") as mock_engine,
        patch("app.agent_orchestrator.process_user_input_stream", return_value=["msg1", "msg2"]),
    ):
        mock_engine.return_value.connect.return_value = mock_db_connection
        result = await stream_sql_mode(args, usage_limits)
    assert result["success"] is True


@pytest.mark.asyncio
async def test_stream_dataframe_mode_csv(mock_df, temp_dir, usage_limits):  # noqa: PT019
    args = MagicMock()
    args.prompt = "test"
    args.file = str(temp_dir / "test.csv")
    args.sheet = None
    with (
        patch("pandas.read_csv", return_value=mock_df),
        patch("app.agent_orchestrator.process_user_input_stream", return_value=["msg1", "msg2"]),
    ):
        result = await stream_dataframe_mode(args, usage_limits)
    assert result["success"] is True


def test_display_results_success(temp_dir):
    result = {
        "message": "Done",
        "sql_query": "SELECT *",
        "visualization_path": str(temp_dir / "vis.html"),
        "data_summary": {"shape": (3, 2), "columns": ["A", "B"]},
    }
    with patch("builtins.print") as mock_print, patch("webbrowser.open"):
        display_results(result)
    mock_print.assert_called()


def test_display_results_error():
    result = {"error": "Test error"}
    with patch("builtins.print") as mock_print:
        display_results(result)
    mock_print.assert_called_with("\n--- Error ---\nTest error")


@pytest.mark.asyncio
async def test_process_query_sql_success(mock_db_connection):
    history = []
    with (
        patch(
            "app.agent_orchestrator.process_user_input_stream",
            return_value=["Starting analysis...", "Visualization saved to: vis.html"],
        ),
        patch("app.visualization_server.serve_visualization", return_value="http://localhost:8081/vis.html"),
    ):
        async for h, _viz in process_query(history, "test", "sql", None, 4000, 10, db_connection=mock_db_connection):
            history = h
    assert any("Visualization is ready" in msg.content for msg in history)
