/** Types shared across the chat panel. */

export interface AgentMessage {
  id: string;
  role: "user" | "agent";
  content: string;
  timestamp: number;
}

export interface ToolEvent {
  id: string;
  type: "executing" | "completed" | "failed";
  toolName: string;
  result?: string;
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
  | { kind: "tool"; data: ToolEvent }
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
