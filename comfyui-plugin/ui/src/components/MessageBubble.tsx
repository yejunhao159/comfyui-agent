import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { AgentMessage } from "../types";

interface Props {
  message: AgentMessage;
  isStreaming: boolean;
}

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
              // Open links in new tab
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
    </div>
  );
};
