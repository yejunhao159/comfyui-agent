import React from "react";
import type { ConnectionStatus } from "../types";

interface Props {
  status: ConnectionStatus;
  onClear: () => void;
}

export const Header: React.FC<Props> = ({ status, onClear }) => {
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
        <span className="cua-header-title">ComfyUI Agent</span>
        <span className="cua-header-status">
          <span className="cua-dot" style={{ background: color }} />
          {label}
        </span>
      </div>
      <button className="cua-header-clear" onClick={onClear} title="清空对话">
        🗑
      </button>
    </div>
  );
};
