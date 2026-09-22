"""Tests for CAO_MCP_ALLOWED_TOOLS role-aware tool filtering."""

import asyncio
from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from cli_agent_orchestrator.mcp_server.server import (
    ALL_TOOL_NAMES,
    apply_tool_allowlist,
    mcp,
)


class TestApplyToolAllowlistComputation:
    """Unit tests for apply_tool_allowlist()'s disable-set computation, mocking mcp.disable
    so these never mutate the real shared server instance."""

    @patch("cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS", "")
    @patch("cli_agent_orchestrator.mcp_server.server.mcp")
    def test_empty_var_is_a_noop(self, mock_mcp):
        """Unset/empty CAO_MCP_ALLOWED_TOOLS must never call disable() — full pass-through."""
        apply_tool_allowlist()
        mock_mcp.disable.assert_not_called()

    @patch(
        "cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS",
        "send_message,load_skill",
    )
    @patch("cli_agent_orchestrator.mcp_server.server.mcp")
    def test_non_empty_var_disables_the_complement(self, mock_mcp):
        """A non-empty allowlist disables every registered tool NOT in that list."""
        apply_tool_allowlist()
        mock_mcp.disable.assert_called_once()
        _, kwargs = mock_mcp.disable.call_args
        assert kwargs["names"] == ALL_TOOL_NAMES - {"send_message", "load_skill"}
        assert kwargs["components"] == {"tool"}

    @patch(
        "cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS",
        " send_message , load_skill ,, ",
    )
    @patch("cli_agent_orchestrator.mcp_server.server.mcp")
    def test_whitespace_and_empty_entries_are_tolerated(self, mock_mcp):
        """Surrounding whitespace and stray empty entries (trailing comma, double comma) must
        not produce a phantom allowed-tool name or crash the parse."""
        apply_tool_allowlist()
        _, kwargs = mock_mcp.disable.call_args
        assert kwargs["names"] == ALL_TOOL_NAMES - {"send_message", "load_skill"}

    @patch(
        "cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS",
        ",".join(sorted(ALL_TOOL_NAMES)),
    )
    @patch("cli_agent_orchestrator.mcp_server.server.mcp")
    def test_allowlist_naming_every_tool_disables_nothing(self, mock_mcp):
        """An explicit allowlist that happens to name every real tool disables an empty set —
        disable() must not be called with an empty names set (a harmless no-op call, but the
        cleaner behavior is to skip it entirely)."""
        apply_tool_allowlist()
        mock_mcp.disable.assert_not_called()


class TestToolAllowlistIntegration:
    """End-to-end behavioral tests against the real global `mcp` instance via an in-memory
    FastMCP Client. Each test restores full visibility in a finally block so later tests (in
    this file or elsewhere in the suite) see every tool enabled, matching this server's normal
    default state."""

    def _restore_all_enabled(self):
        mcp.enable(names=ALL_TOOL_NAMES, components={"tool"})

    @pytest.mark.asyncio
    async def test_filtered_tool_list_and_call_rejection(self):
        with patch(
            "cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS",
            "send_message,load_skill",
        ):
            apply_tool_allowlist()
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                assert names == {"send_message", "load_skill"}

                with pytest.raises(ToolError):
                    await client.call_tool("handoff", {"agent_profile": "developer", "message": "x"})
        finally:
            self._restore_all_enabled()

    @pytest.mark.asyncio
    async def test_allowed_call_still_reaches_the_real_tool(self):
        """A tool call within the allowlist is dispatched normally — filtering must not
        interfere with the tools it does not disable. load_skill's own network dependency
        (_load_skill_impl) is mocked so this test proves the call reached the real handler
        without depending on a live cao-server."""
        with patch(
            "cli_agent_orchestrator.mcp_server.server.CAO_MCP_ALLOWED_TOOLS",
            "load_skill",
        ):
            apply_tool_allowlist()
        try:
            with patch(
                "cli_agent_orchestrator.mcp_server.server._load_skill_impl",
                return_value="mock skill content",
            ) as mock_impl:
                async with Client(mcp) as client:
                    result = await client.call_tool("load_skill", {"name": "some-skill"})
            mock_impl.assert_called_once_with("some-skill")
            assert result.content[0].text == "mock skill content"
        finally:
            self._restore_all_enabled()


class TestAllToolNamesDriftGuard:
    """Guards against ALL_TOOL_NAMES silently drifting out of sync with the tools actually
    registered via @mcp.tool() — the runtime filtering logic reads this constant, not a live
    introspection of the server, so a forgotten update here would let a newly-added tool bypass
    filtering entirely (always allowed) or a removed tool linger in every disable-set
    computation (harmless, but a sign the constant is stale)."""

    def test_all_tool_names_matches_live_registration(self):
        tools = asyncio.run(mcp.list_tools())
        assert {t.name for t in tools} == ALL_TOOL_NAMES
