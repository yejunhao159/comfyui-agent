import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ChatItem,
  ConnectionStatus,
  ServerEvent,
  TurnStats,
} from "../types";

const RECONNECT_DELAY = 3000;

/** Hook that manages the WebSocket connection to the Python Agent. */
export function useAgentWS(agentUrl: string) {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const streamTextRef = useRef("");
  const itemIdRef = useRef(0);

  const nextId = () => String(++itemIdRef.current);

  // --- helpers to update items ---
  const appendItem = useCallback((item: ChatItem) => {
    setItems((prev) => [...prev, item]);
  }, []);

  const updateLastAgentMessage = useCallback((content: string) => {
    setItems((prev) => {
      const copy = [...prev];
      for (let i = copy.length - 1; i >= 0; i--) {
        const it = copy[i];
        if (it?.kind === "message" && it.data.role === "agent") {
          copy[i] = {
            ...it,
            data: { ...it.data, content },
          };
          break;
        }
      }
      return copy;
    });
  }, []);

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
          if (msg.session_id) setSessionId(msg.session_id);
          break;

        case "event":
          handleEvent(msg);
          break;

        case "response":
          // Final response — update the streaming message
          if (msg.content) {
            updateLastAgentMessage(msg.content);
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
              timestamp: Date.now(),
            },
          });
          setIsProcessing(false);
          break;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [appendItem, updateLastAgentMessage]
  );

  const handleEvent = useCallback(
    (msg: ServerEvent) => {
      const et = msg.event_type ?? "";
      const data = msg.data ?? {};

      if (et === "stream.text_delta") {
        streamTextRef.current += (data.text as string) ?? "";
        updateLastAgentMessage(streamTextRef.current);
      } else if (et === "state.conversation_start") {
        streamTextRef.current = "";
        // Create a placeholder agent message for streaming
        appendItem({
          kind: "message",
          data: {
            id: nextId(),
            role: "agent",
            content: "",
            timestamp: Date.now(),
          },
        });
      } else if (
        et === "state.tool_executing" ||
        et === "state.tool_completed" ||
        et === "state.tool_failed"
      ) {
        const typeMap: Record<string, "executing" | "completed" | "failed"> = {
          "state.tool_executing": "executing",
          "state.tool_completed": "completed",
          "state.tool_failed": "failed",
        };
        appendItem({
          kind: "tool",
          data: {
            id: nextId(),
            type: typeMap[et]!,
            toolName: (data.tool_name as string) ?? "unknown",
            result: data.result as string | undefined,
            timestamp: Date.now(),
          },
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
      }
    },
    [appendItem, updateLastAgentMessage]
  );

  // --- send message ---
  const sendMessage = useCallback(
    (text: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      // Add user message
      appendItem({
        kind: "message",
        data: {
          id: nextId(),
          role: "user",
          content: text,
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

  return {
    status,
    items,
    sessionId,
    isProcessing,
    sendMessage,
    cancelRequest,
    clearItems: () => setItems([]),
  };
}
