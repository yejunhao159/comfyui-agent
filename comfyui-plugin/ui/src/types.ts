/** Types shared across the chat panel. */

export interface ToolCall {
  id: string;
  name: string;
  status: "executing" | "completed" | "failed";
  result?: string;
  error?: string;
}

export interface SubAgentBlock {
  id: string;
  task: string;
  status: "executing" | "completed" | "failed";
  result?: string;
}

export interface RetryNotice {
  attempt: number;
  maxRetries: number;
  delayMs: number;
  error: string;
}

export interface ExperienceNotice {
  name: string;
  title: string;
}

/** Ordered content block — text, tool call, sub-agent, or retry notice. */
export type ContentBlock =
  | { kind: "text"; text: string }
  | { kind: "tool"; tool: ToolCall }
  | { kind: "subagent"; subagent: SubAgentBlock }
  | { kind: "retry"; retry: RetryNotice };

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
  | { kind: "stats"; data: TurnStats }
  | { kind: "experience"; data: ExperienceNotice };

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

/** Session metadata from the backend. */
export interface Session {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
}
