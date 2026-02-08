import React from "react";
import type { ConnectionStatus } from "../types";

interface Props {
  status: ConnectionStatus;
  onClear: () => void;
  onToggleSessions?: () => void;
}

export const Header: React.FC<Props> = ({ status, onClear, onToggleSessions }) => {
  const statusConfig: Record<
    ConnectionStatus,
    { color: string; label: string }
  > = {
    connecting: { color: "#FBBF24", label: "连接中..." },
    connected: { color: "#34D399", label: "已连接" },
    disconnected: { color: "#EF4444", label: "已断开" },
  };

  const { color, label } = statusConfig[status];

  return (
    <div className="cua-header">
      <div className="cua-header-left">
        {onToggleSessions && (
          <button
            className="cua-header-btn"
            onClick={onToggleSessions}
            title="会话列表"
          >
            ☰
          </button>
        )}
        <span className="cua-header-title">ComfyUI Agent</span>
        <span className="cua-header-status">
          <span className="cua-dot" style={{ background: color }} />
          {label}
        </span>
      </div>
      <button className="cua-header-btn" onClick={onClear} title="新建会话">
        +
      </button>
    </div>
  );
};
