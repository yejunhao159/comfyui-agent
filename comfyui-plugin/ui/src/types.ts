/** Types shared across the chat panel. */

export interface ToolCall {
  id: string;
  name: string;
  status: "executing" | "completed" | "failed";
  result?: string;
  error?: string;
}

/** Ordered content block — text or tool call, rendered chronologically. */
export type ContentBlock =
  | { kind: "text"; text: string }
  | { kind: "tool"; tool: ToolCall };

export interface AgentMessage {
  id: string;
  role: "user" | "agent";
  content: string;
  toolCalls: ToolCall[];
  blocks: ContentBlock[];
  timestamp: number;
}

export interface TurnStats {
  duration: number;
  iterations: number;
  inputTokens: number;
  outputTokens: number;
}

export type ChatItem =
  | { kind: "message"; data: AgentMessage }
  | { kind: "stats"; data: TurnStats };

/** WebSocket message from the server. */
export interface ServerEvent {
  type: "event" | "response" | "error" | "pong" | "session_created";
  event_type?: string;
  data?: Record<string, unknown>;
  session_id?: string;
  content?: string;
  error?: string;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected";
