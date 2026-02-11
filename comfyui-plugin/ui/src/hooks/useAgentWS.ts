import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ChatItem,
  ConnectionStatus,
  ContentBlock,
  ExperienceNotice,
  RetryNotice,
  ServerEvent,
  SubAgentBlock,
  ToolCall,
  TurnStats,
} from "../types";
import { getSavedSessionId, saveSessionId } from "./useSessionAPI";

const RECONNECT_DELAY = 3000;

/** Hook that manages the WebSocket connection to the Python Agent. */
export function useAgentWS(agentUrl: string) {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(
    getSavedSessionId()
  );
  const [isProcessing, setIsProcessing] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const streamTextRef = useRef("");
  const itemIdRef = useRef(0);

  const nextId = () => String(++itemIdRef.current);

  // --- helpers to update items ---
  const appendItem = useCallback((item: ChatItem) => {
    setItems((prev) => [...prev, item]);
  }, []);

  /** Update the last agent message's blocks array. */
  const updateLastAgent = useCallback(
    (updater: (msg: ChatItem & { kind: "message" }) => ChatItem) => {
      setItems((prev) => {
        const copy = [...prev];
        for (let i = copy.length - 1; i >= 0; i--) {
          const it = copy[i]!;
          if (it.kind === "message" && it.data.role === "agent") {
            copy[i] = updater(it as ChatItem & { kind: "message" });
            break;
          }
        }
        return copy;
      });
    },
    []
  );

  /** Append or update the last text block in the agent message's blocks. */
  const updateStreamingText = useCallback(
    (text: string) => {
      updateLastAgent((it) => {
        const blocks = [...it.data.blocks];
        const last = blocks[blocks.length - 1];
        if (last && last.kind === "text") {
          blocks[blocks.length - 1] = { kind: "text", text };
        } else {
          blocks.push({ kind: "text", text });
        }
        return {
          ...it,
          data: { ...it.data, content: text, blocks },
        };
      });
    },
    [updateLastAgent]
  );

  /** Push a new tool block into the agent message's blocks. */
  const pushToolBlock = useCallback(
    (tool: ToolCall) => {
      updateLastAgent((it) => {
        const toolCalls = [...it.data.toolCalls, tool];
        const blocks: ContentBlock[] = [
          ...it.data.blocks,
          { kind: "tool", tool },
        ];
        return {
          ...it,
          data: { ...it.data, toolCalls, blocks },
        };
      });
    },
    [updateLastAgent]
  );

  /** Update a tool by name in both toolCalls and blocks. */
  const updateTool = useCallback(
    (toolName: string, patch: Partial<ToolCall>) => {
      updateLastAgent((it) => {
        // Update toolCalls
        const toolCalls = [...it.data.toolCalls];
        for (let i = toolCalls.length - 1; i >= 0; i--) {
          const tc = toolCalls[i]!;
          if (tc.name === toolName && (patch.status ? tc.status === "executing" : true)) {
            toolCalls[i] = { ...tc, ...patch };
            break;
          }
        }
        // Update blocks
        const blocks = [...it.data.blocks];
        for (let i = blocks.length - 1; i >= 0; i--) {
          const b = blocks[i]!;
          if (b.kind === "tool" && b.tool.name === toolName &&
              (patch.status ? b.tool.status === "executing" : true)) {
            blocks[i] = { kind: "tool", tool: { ...b.tool, ...patch } };
            break;
          }
        }
        return {
          ...it,
          data: { ...it.data, toolCalls, blocks },
        };
      });
    },
    [updateLastAgent]
  );

  /** Push a sub-agent block into the agent message's blocks. */
  const pushSubAgentBlock = useCallback(
    (subagent: SubAgentBlock) => {
      updateLastAgent((it) => {
        const blocks: ContentBlock[] = [
          ...it.data.blocks,
          { kind: "subagent", subagent },
        ];
        return { ...it, data: { ...it.data, blocks } };
      });
    },
    [updateLastAgent]
  );

  /** Push a retry notice block. */
  const pushRetryBlock = useCallback(
    (retry: RetryNotice) => {
      updateLastAgent((it) => {
        const blocks: ContentBlock[] = [
          ...it.data.blocks,
          { kind: "retry", retry },
        ];
        return { ...it, data: { ...it.data, blocks } };
      });
    },
    [updateLastAgent]
  );

  // --- WebSocket connection ---
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(agentUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      ws.send(JSON.stringify({ type: "ping" }));
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      setStatus("disconnected");
    };

    ws.onmessage = (e) => {
      try {
        const msg: ServerEvent = JSON.parse(e.data as string);
        handleServerMessage(msg);
      } catch {
        // ignore parse errors
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentUrl]);

  const handleServerMessage = useCallback(
    (msg: ServerEvent) => {
      switch (msg.type) {
        case "pong":
          break;

        case "session_created":
          if (msg.session_id) {
            setSessionId(msg.session_id);
            saveSessionId(msg.session_id);
          }
          break;

        case "event":
          handleEvent(msg);
          break;

        case "response":
          if (msg.content) {
            // Final content — update the last text block
            updateStreamingText(msg.content);
          }
          streamTextRef.current = "";
          setIsProcessing(false);
          break;

        case "error":
          appendItem({
            kind: "message",
            data: {
              id: nextId(),
              role: "agent",
              content: `**Error:** ${msg.error ?? "Unknown error"}`,
              toolCalls: [],
              blocks: [{ kind: "text", text: `**Error:** ${msg.error ?? "Unknown error"}` }],
              timestamp: Date.now(),
            },
          });
          setIsProcessing(false);
          break;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [appendItem, updateStreamingText]
  );

  const handleEvent = useCallback(
    (msg: ServerEvent) => {
      const et = msg.event_type ?? "";
      const data = msg.data ?? {};

      if (et === "stream.text_delta") {
        streamTextRef.current += (data.text as string) ?? "";
        updateStreamingText(streamTextRef.current);
      } else if (et === "state.thinking") {
        // New LLM iteration — reset stream so next text becomes a new block
        streamTextRef.current = "";
      } else if (et === "state.conversation_start") {
        streamTextRef.current = "";
        appendItem({
          kind: "message",
          data: {
            id: nextId(),
            role: "agent",
            content: "",
            toolCalls: [],
            blocks: [],
            timestamp: Date.now(),
          },
        });
      } else if (et === "state.tool_executing") {
        const toolName = (data.tool_name as string) ?? "unknown";
        const toolId = (data.tool_id as string) ?? nextId();
        pushToolBlock({ id: toolId, name: toolName, status: "executing" });
      } else if (et === "state.tool_completed") {
        const toolName = (data.tool_name as string) ?? "";
        updateTool(toolName, { status: "completed" });
      } else if (et === "state.tool_failed") {
        const toolName = (data.tool_name as string) ?? "";
        const error = (data.error as string) ?? "Unknown error";
        updateTool(toolName, { status: "failed", error });
      } else if (et === "message.tool_result") {
        const toolName = (data.tool_name as string) ?? "";
        const result = (data.result as string) ?? "";
        updateTool(toolName, { result });
      } else if (et === "subagent.start") {
        const childId = (data.child_session_id as string) ?? nextId();
        const task = (data.task as string) ?? "";
        pushSubAgentBlock({ id: childId, task, status: "executing" });
      } else if (et === "subagent.end") {
        // Find the executing sub-agent block and mark it completed
        const preview = (data.result_preview as string) ?? "";
        updateLastAgent((it) => {
          const blocks = [...it.data.blocks];
          for (let i = blocks.length - 1; i >= 0; i--) {
            const b = blocks[i]!;
            if (b.kind === "subagent" && b.subagent.status === "executing") {
              const failed = preview.startsWith("Error:");
              blocks[i] = {
                kind: "subagent",
                subagent: {
                  ...b.subagent,
                  status: failed ? "failed" : "completed",
                  result: preview,
                },
              };
              break;
            }
          }
          return { ...it, data: { ...it.data, blocks } };
        });
      } else if (et === "llm.retry") {
        pushRetryBlock({
          attempt: (data.attempt as number) ?? 0,
          maxRetries: (data.max_retries as number) ?? 0,
          delayMs: (data.delay_ms as number) ?? 0,
          error: (data.error as string) ?? "",
        });
      } else if (et === "turn.end") {
        const usage = (data.usage as Record<string, number>) ?? {};
        const stats: TurnStats = {
          duration: (data.duration as number) ?? 0,
          iterations: (data.iterations as number) ?? 0,
          inputTokens: usage.input_tokens ?? 0,
          outputTokens: usage.output_tokens ?? 0,
        };
        appendItem({ kind: "stats", data: stats });
      } else if (et === "workflow.submitted") {
        window.dispatchEvent(
          new CustomEvent("comfyui-agent:load-workflow", {
            detail: {
              workflow: data.workflow,
              promptId: data.prompt_id,
            },
          })
        );
      } else if (et === "experience.synthesized") {
        const notice: ExperienceNotice = {
          name: (data.name as string) ?? "",
          title: (data.title as string) ?? "经验升级",
        };
        appendItem({ kind: "experience", data: notice });
      }
    },
    [appendItem, updateStreamingText, pushToolBlock, updateTool, pushSubAgentBlock, updateLastAgent, pushRetryBlock]
  );

  // --- send message ---
  const sendMessage = useCallback(
    (text: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      appendItem({
        kind: "message",
        data: {
          id: nextId(),
          role: "user",
          content: text,
          toolCalls: [],
          blocks: [],
          timestamp: Date.now(),
        },
      });

      setIsProcessing(true);

      wsRef.current.send(
        JSON.stringify({
          type: "chat",
          session_id: sessionId,
          message: text,
        })
      );
    },
    [sessionId, appendItem]
  );

  const cancelRequest = useCallback(() => {
    if (!wsRef.current || !sessionId) return;
    wsRef.current.send(
      JSON.stringify({ type: "cancel", session_id: sessionId })
    );
    setIsProcessing(false);
  }, [sessionId]);

  // --- lifecycle ---
  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  // Listen for "ask about node" events from the ComfyUI bridge
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        nodeType: string;
        nodeTitle: string;
      };
      if (detail?.nodeType) {
        sendMessage(
          `Tell me about the "${detail.nodeTitle}" node (type: ${detail.nodeType}). What does it do and what are its inputs/outputs?`
        );
      }
    };
    window.addEventListener("comfyui-agent:ask-node", handler);
    return () => window.removeEventListener("comfyui-agent:ask-node", handler);
  }, [sendMessage]);

  /** Switch to a different session (called from session list). */
  const switchSession = useCallback(
    (newSessionId: string, historyItems: ChatItem[]) => {
      setSessionId(newSessionId);
      saveSessionId(newSessionId);
      setItems(historyItems);
      streamTextRef.current = "";
      setIsProcessing(false);
    },
    []
  );

  /** Start a new empty session. */
  const newSession = useCallback(() => {
    setSessionId(null);
    setItems([]);
    streamTextRef.current = "";
    setIsProcessing(false);
    // Don't save to localStorage — will be saved when server creates it
  }, []);

  return {
    status,
    items,
    sessionId,
    isProcessing,
    sendMessage,
    cancelRequest,
    switchSession,
    newSession,
    clearItems: () => setItems([]),
  };
}
