"""Interactive CLI for the ComfyUI Agent.

Provides a natural conversation experience with:
- Streaming LLM responses (text appears character by character)
- Tool call status indicators (spinners, progress)
- Formatted messages (user/assistant/tool with different styles)
- Logs separated from conversation
"""

from __future__ import annotations

import asyncio
import logging
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from comfyui_agent.application.agent_loop import AgentLoop
from comfyui_agent.domain.models.events import Event, EventType
from comfyui_agent.domain.tools.factory import create_all_tools
from comfyui_agent.infrastructure.clients.comfyui_client import ComfyUIClient
from comfyui_agent.infrastructure.config import AppConfig
from comfyui_agent.infrastructure.event_bus import EventBus
from comfyui_agent.infrastructure.clients.llm_client import LLMClient
from comfyui_agent.infrastructure.persistence.session_store import SessionStore

logger = logging.getLogger(__name__)

console = Console()


class CLIRenderer:
    """Renders agent events to the terminal in a natural conversation style."""

    def __init__(self) -> None:
        self._streaming_text = ""
        self._live: Live | None = None
        self._current_tool: str | None = None

    def print_welcome(self, comfyui_ok: bool) -> None:
        console.print()
        console.print("[bold cyan]ComfyUI Agent[/bold cyan]", justify="center")
        console.print("[dim]Natural language control for ComfyUI[/dim]", justify="center")
        console.print()
        if comfyui_ok:
            console.print("  [green]●[/green] ComfyUI connected")
        else:
            console.print("  [red]●[/red] ComfyUI not reachable")
        console.print("  [dim]Type your message, or 'quit' to exit[/dim]")
        console.print()

    def print_user_message(self, text: str) -> None:
        console.print()
        console.print(f"[bold blue]You:[/bold blue] {text}")

    def start_streaming(self) -> None:
        self._streaming_text = ""
        self._live = Live(
            Text("", style="dim"),
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )
        self._live.start()

    def stream_text(self, delta: str) -> None:
        self._streaming_text += delta
        if self._live:
            self._live.update(Markdown(self._streaming_text))

    def stop_streaming(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def print_assistant_message(self, text: str) -> None:
        self.stop_streaming()
        console.print()
        console.print("[bold green]Agent:[/bold green]")
        console.print(Markdown(text))

    def print_tool_start(self, tool_name: str) -> None:
        self.stop_streaming()
        self._current_tool = tool_name
        display_name = tool_name.replace("comfyui_", "").replace("_", " ").title()
        console.print(f"\n  [yellow]⚡[/yellow] [dim]{display_name}...[/dim]", end="")

    def print_tool_result(self, tool_name: str, is_error: bool) -> None:
        if is_error:
            console.print(f" [red]✗[/red]")
        else:
            console.print(f" [green]✓[/green]")
        self._current_tool = None

    def print_tool_detail(self, text: str) -> None:
        """Show abbreviated tool output."""
        lines = text.strip().split("\n")
        if len(lines) <= 5:
            for line in lines:
                console.print(f"    [dim]{line}[/dim]")
        else:
            for line in lines[:3]:
                console.print(f"    [dim]{line}[/dim]")
            console.print(f"    [dim]... ({len(lines) - 3} more lines)[/dim]")

    def print_turn_stats(self, duration: float, iterations: int, usage: dict) -> None:
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        console.print(
            f"\n  [dim]({duration:.1f}s · {iterations} step{'s' if iterations > 1 else ''} · {tokens} tokens)[/dim]"
        )

    def print_error(self, error: str) -> None:
        self.stop_streaming()
        console.print(f"\n[bold red]Error:[/bold red] {error}")


def setup_event_handlers(event_bus: EventBus, renderer: CLIRenderer) -> None:
    """Wire event bus to CLI renderer for real-time display."""

    async def on_text_delta(event: Event) -> None:
        renderer.stream_text(event.data.get("text", ""))

    async def on_tool_executing(event: Event) -> None:
        renderer.print_tool_start(event.data.get("tool_name", "unknown"))

    async def on_tool_completed(event: Event) -> None:
        renderer.print_tool_result(event.data.get("tool_name", ""), is_error=False)

    async def on_tool_failed(event: Event) -> None:
        renderer.print_tool_result(event.data.get("tool_name", ""), is_error=True)

    async def on_tool_result(event: Event) -> None:
        result_text = event.data.get("result", "")
        if result_text:
            renderer.print_tool_detail(result_text)

    async def on_turn_end(event: Event) -> None:
        renderer.print_turn_stats(
            event.data.get("duration", 0),
            event.data.get("iterations", 0),
            event.data.get("usage", {}),
        )

    async def on_conversation_start(event: Event) -> None:
        renderer.start_streaming()

    event_bus.on(EventType.STREAM_TEXT_DELTA, on_text_delta)
    event_bus.on(EventType.STATE_TOOL_EXECUTING, on_tool_executing)
    event_bus.on(EventType.STATE_TOOL_COMPLETED, on_tool_completed)
    event_bus.on(EventType.STATE_TOOL_FAILED, on_tool_failed)
    event_bus.on(EventType.MESSAGE_TOOL_RESULT, on_tool_result)
    event_bus.on(EventType.TURN_END, on_turn_end)
    event_bus.on(EventType.STATE_CONVERSATION_START, on_conversation_start)


async def run_cli() -> None:
    """Main CLI entry point."""
    config = AppConfig.from_yaml()

    # Setup logging — file only, not terminal
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("data/agent.log", mode="a")],
    )

    # Initialize components
    event_bus = EventBus()
    renderer = CLIRenderer()
    setup_event_handlers(event_bus, renderer)

    comfyui = ComfyUIClient(
        base_url=config.comfyui.base_url,
        ws_url=config.comfyui.ws_url,
        timeout=config.comfyui.timeout,
        event_bus=event_bus,
    )

    api_key = config.llm.resolve_api_key()
    if not api_key:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set")
        console.print("Set it via environment variable or in config.yaml")
        return

    llm = LLMClient(
        api_key=api_key,
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
        event_bus=event_bus,
    )

    session_store = SessionStore(db_path=config.agent.session_db)

    # Build node index
    from comfyui_agent.knowledge.node_index import NodeIndex
    node_index = NodeIndex()

    tools = create_all_tools(comfyui, node_index)

    agent = AgentLoop(
        llm=llm,
        tools=tools,
        session_store=session_store,
        event_bus=event_bus,
        max_iterations=config.agent.max_iterations,
    )

    # Health check
    comfyui_ok = await comfyui.health_check()
    renderer.print_welcome(comfyui_ok)

    if comfyui_ok:
        await comfyui.connect_ws()
        await node_index.build(comfyui)

    # Create session
    session_id = await session_store.create_session("CLI Session")

    # Interactive loop
    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory("data/.cli_history"),
    )

    try:
        while True:
            try:
                with patch_stdout():
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: prompt_session.prompt("→ "),
                    )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            renderer.print_user_message(user_input)

            try:
                response = await agent.run(session_id, user_input)
                renderer.stop_streaming()
                renderer.print_assistant_message(response)
            except Exception as e:
                renderer.print_error(str(e))

    finally:
        console.print("\n[dim]Goodbye![/dim]")
        await comfyui.close()
        await llm.close()
        await session_store.close()


def main() -> None:
    """Entry point."""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
