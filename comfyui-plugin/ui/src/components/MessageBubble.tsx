import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { AgentMessage, ContentBlock, RetryNotice, SubAgentBlock, ToolCall } from "../types";

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

const SUBAGENT_ICONS: Record<string, string> = {
  executing: "\u{1F50D}",
  completed: "\u2713",
  failed: "\u2717",
};

const SUBAGENT_LABELS: Record<string, string> = {
  executing: "\u7814\u7A76\u4E2D...",
  completed: "\u5B8C\u6210",
  failed: "\u5931\u8D25",
};

const SubAgentItem: React.FC<{ subagent: SubAgentBlock }> = ({ subagent }) => {
  const [open, setOpen] = useState(subagent.status === "executing");
  const prevStatus = useRef(subagent.status);

  useEffect(() => {
    if (prevStatus.current === "executing" && subagent.status !== "executing") {
      // Auto-collapse after a short delay so user can see the result
      const timer = setTimeout(() => setOpen(false), 1500);
      return () => clearTimeout(timer);
    }
    prevStatus.current = subagent.status;
  }, [subagent.status]);

  return (
    <div className={`cua-subagent cua-subagent-${subagent.status}`}>
      <div
        className="cua-subagent-header"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="cua-subagent-icon">{SUBAGENT_ICONS[subagent.status]}</span>
        <span className="cua-subagent-name">delegate_task</span>
        <span className="cua-subagent-badge">{SUBAGENT_LABELS[subagent.status]}</span>
        <span className={`cua-toolcall-chevron ${open ? "cua-chevron-open" : ""}`}>
          {"\u25B8"}
        </span>
      </div>
      <div className={`cua-subagent-body ${open ? "cua-body-open" : ""}`}>
        <div className="cua-subagent-task">
          <strong>{"\u4EFB\u52A1:"}</strong> {subagent.task}
        </div>
        {subagent.status === "executing" && (
          <div className="cua-subagent-progress">
            <span className="cua-thinking">{"\u5B50\u4EE3\u7406\u6B63\u5728\u7814\u7A76\u4E2D..."}</span>
          </div>
        )}
        {subagent.result && (
          <pre className="cua-toolcall-result">{subagent.result}</pre>
        )}
      </div>
    </div>
  );
};

const RetryNoticeItem: React.FC<{ retry: RetryNotice }> = ({ retry }) => (
  <div className="cua-retry">
    <span className="cua-retry-icon">{"\u26A0"}</span>
    <span className="cua-retry-text">
      {`\u91CD\u8BD5\u4E2D (${retry.attempt}/${retry.maxRetries}) \u2014 ${(retry.delayMs / 1000).toFixed(1)}s \u540E\u91CD\u8BD5`}
    </span>
  </div>
);

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

/** Render a single content block. */
const BlockRenderer: React.FC<{
  block: ContentBlock;
  isLast: boolean;
  isStreaming: boolean;
}> = ({ block, isLast, isStreaming }) => {
  if (block.kind === "text") {
    const streaming = isLast && isStreaming;
    return (
      <div className={`cua-msg-content ${streaming ? "cua-streaming" : ""}`}>
        <AgentMarkdown content={block.text} />
      </div>
    );
  }
  if (block.kind === "subagent") {
    return (
      <div className="cua-toolcalls">
        <SubAgentItem subagent={block.subagent} />
      </div>
    );
  }
  if (block.kind === "retry") {
    return <RetryNoticeItem retry={block.retry} />;
  }
  // tool block
  return (
    <div className="cua-toolcalls">
      <ToolCallItem tool={block.tool} />
    </div>
  );
};

export const MessageBubble: React.FC<Props> = ({ message, isStreaming }) => {
  const isUser = message.role === "user";

  // Throttle streaming updates via rAF
  const [renderedBlocks, setRenderedBlocks] = useState(message.blocks);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!isStreaming) {
      setRenderedBlocks(message.blocks);
      return;
    }
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      setRenderedBlocks(message.blocks);
    });
    return () => cancelAnimationFrame(rafRef.current);
  }, [message.blocks, isStreaming]);

  return (
    <div className={`cua-msg-row ${isUser ? "cua-msg-row-user" : "cua-msg-row-agent"}`}>
      {!isUser && (
        <div className="cua-avatar cua-avatar-agent">A</div>
      )}
      <div className={`cua-msg ${isUser ? "cua-msg-user" : "cua-msg-agent"}`}>
        {isUser ? (
          <div className="cua-msg-content">
            <p>{message.content}</p>
          </div>
        ) : renderedBlocks.length > 0 ? (
          renderedBlocks.map((block, i) => (
            <BlockRenderer
              key={i}
              block={block}
              isLast={i === renderedBlocks.length - 1}
              isStreaming={isStreaming}
            />
          ))
        ) : (
          <div className="cua-msg-content">
            <span className="cua-thinking">Thinking...</span>
          </div>
        )}
      </div>
      {isUser && (
        <div className="cua-avatar cua-avatar-user">Y</div>
      )}
    </div>
  );
};
