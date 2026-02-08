import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { AgentMessage, ToolCall } from "../types";

interface Props {
  message: AgentMessage;
  isStreaming: boolean;
}

const ICONS: Record<string, string> = {
  executing: "\u26A1",
  completed: "\u2713",
  failed: "\u2717",
};

const LABELS: Record<string, string> = {
  executing: "\u6267\u884C\u4E2D...",
  completed: "\u5B8C\u6210",
  failed: "\u5931\u8D25",
};

const ToolCallItem: React.FC<{ tool: ToolCall }> = ({ tool }) => {
  const [open, setOpen] = useState(tool.status === "executing");

  const displayName = tool.name
    .replace("comfyui_", "")
    .replace(/_/g, " ");

  return (
    <div className={`cua-toolcall cua-toolcall-${tool.status}`}>
      <div
        className="cua-toolcall-header"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="cua-toolcall-icon">{ICONS[tool.status]}</span>
        <span className="cua-toolcall-name">{displayName}</span>
        <span className="cua-toolcall-badge">{LABELS[tool.status]}</span>
      </div>
      {open && (
        <div className="cua-toolcall-body">
          {tool.status === "executing" && !tool.result && (
            <span className="cua-thinking">执行中...</span>
          )}
          {tool.result && (
            <pre className="cua-toolcall-result">{tool.result}</pre>
          )}
          {tool.error && (
            <pre className="cua-toolcall-result cua-toolcall-error">
              {tool.error}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};

export const MessageBubble: React.FC<Props> = ({ message, isStreaming }) => {
  const isUser = message.role === "user";

  return (
    <div className={`cua-msg ${isUser ? "cua-msg-user" : "cua-msg-agent"}`}>
      <div className="cua-msg-label">
        {isUser ? "You" : "Agent"}
      </div>
      <div className={`cua-msg-content ${isStreaming ? "cua-streaming" : ""}`}>
        {isUser ? (
          <p>{message.content}</p>
        ) : message.content ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              a: ({ children, ...props }) => (
                <a {...props} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        ) : (
          <span className="cua-thinking">Thinking...</span>
        )}
      </div>
      {!isUser && message.toolCalls.length > 0 && (
        <div className="cua-toolcalls">
          {message.toolCalls.map((tc) => (
            <ToolCallItem key={tc.id} tool={tc} />
          ))}
        </div>
      )}
    </div>
  );
};
