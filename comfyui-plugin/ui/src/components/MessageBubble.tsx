import React, { useEffect, useRef, useState } from "react";
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
  const prevStatus = useRef(tool.status);

  useEffect(() => {
    if (
      prevStatus.current === "executing" &&
      tool.status !== "executing"
    ) {
      setOpen(false);
    }
    prevStatus.current = tool.status;
  }, [tool.status]);

  const displayName = tool.name
    .replace("comfyui_", "")
    .replace(/_/g, " ");

  const hasBody = tool.status === "executing" || tool.result || tool.error;

  return (
    <div className={`cua-toolcall cua-toolcall-${tool.status}`}>
      <div
        className="cua-toolcall-header"
        onClick={() => hasBody && setOpen((v) => !v)}
      >
        <span className="cua-toolcall-icon">{ICONS[tool.status]}</span>
        <span className="cua-toolcall-name">{displayName}</span>
        <span className="cua-toolcall-badge">{LABELS[tool.status]}</span>
        {hasBody && (
          <span className={`cua-toolcall-chevron ${open ? "cua-chevron-open" : ""}`}>
            {"\u25B8"}
          </span>
        )}
      </div>
      <div className={`cua-toolcall-body ${open ? "cua-body-open" : ""}`}>
        {tool.status === "executing" && !tool.result && (
          <span className="cua-thinking">{"\u6267\u884C\u4E2D..."}</span>
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
    </div>
  );
};

const AgentMarkdown = React.memo<{ content: string }>(({ content }) => (
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
    {content}
  </ReactMarkdown>
));

export const MessageBubble: React.FC<Props> = ({ message, isStreaming }) => {
  const isUser = message.role === "user";

  const [rendered, setRendered] = useState(message.content);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setRendered(message.content);
      return;
    }
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      setRendered(message.content);
    });
    return () => cancelAnimationFrame(rafRef.current);
  }, [message.content, isStreaming]);

  const toolCalls = message.toolCalls;

  return (
    <div className={`cua-msg-row ${isUser ? "cua-msg-row-user" : "cua-msg-row-agent"}`}>
      {!isUser && (
        <div className="cua-avatar cua-avatar-agent">A</div>
      )}
      <div className={`cua-msg ${isUser ? "cua-msg-user" : "cua-msg-agent"}`}>
        <div className={`cua-msg-content ${isStreaming ? "cua-streaming" : ""}`}>
          {isUser ? (
            <p>{message.content}</p>
          ) : rendered ? (
            <AgentMarkdown content={rendered} />
          ) : toolCalls.length === 0 ? (
            <span className="cua-thinking">Thinking...</span>
          ) : null}
        </div>
        {!isUser && toolCalls.length > 0 && (
          <div className="cua-toolcalls">
            {toolCalls.map((tc) => (
              <ToolCallItem key={tc.id} tool={tc} />
            ))}
          </div>
        )}
      </div>
      {isUser && (
        <div className="cua-avatar cua-avatar-user">Y</div>
      )}
    </div>
  );
};
