import React from "react";
import type { ToolEvent } from "../types";

interface Props {
  event: ToolEvent;
}

const ICONS: Record<string, string> = {
  executing: "⚡",
  completed: "✓",
  failed: "✗",
};

export const ToolStatus: React.FC<Props> = ({ event }) => {
  const displayName = event.toolName
    .replace("comfyui_", "")
    .replace(/_/g, " ");

  const labels: Record<string, string> = {
    executing: "执行中...",
    completed: "完成",
    failed: "失败",
  };

  return (
    <div className={`cua-tool cua-tool-${event.type}`}>
      <span className="cua-tool-icon">{ICONS[event.type]}</span>
      <span className="cua-tool-name">{displayName}</span>
      <span className="cua-tool-label">{labels[event.type]}</span>
    </div>
  );
};
