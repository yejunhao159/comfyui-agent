import React, { useEffect, useRef } from "react";
import type { ChatItem } from "../types";
import { MessageBubble } from "./MessageBubble";
import { ExperienceNoticeBar } from "./ExperienceNoticeBar";
import { TurnStatsBar } from "./TurnStatsBar";

interface Props {
  items: ChatItem[];
  isStreaming: boolean;
}

export const MessageList: React.FC<Props> = ({ items, isStreaming }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);

  if (items.length === 0) {
    return (
      <div className="cua-empty">
        <div className="cua-empty-icon">💬</div>
        <p>Ask me anything about ComfyUI</p>
        <p className="cua-empty-hint">
          I can help you build workflows, explain nodes, and generate images.
        </p>
      </div>
    );
  }

  return (
    <div className="cua-messages">
      {items.map((item, i) => {
        switch (item.kind) {
          case "message":
            return (
              <MessageBubble
                key={item.data.id}
                message={item.data}
                isStreaming={
                  isStreaming &&
                  item.data.role === "agent" &&
                  i === items.length - 1
                }
              />
            );
          case "stats":
            return <TurnStatsBar key={`stats-${i}`} stats={item.data} />;
          case "experience":
            return <ExperienceNoticeBar key={`exp-${i}`} notice={item.data} />;
        }
      })}
      <div ref={bottomRef} />
    </div>
  );
};
