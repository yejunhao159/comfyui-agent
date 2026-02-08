import React from "react";
import type { TurnStats } from "../types";

interface Props {
  stats: TurnStats;
}

export const TurnStatsBar: React.FC<Props> = ({ stats }) => {
  const totalTokens = stats.inputTokens + stats.outputTokens;
  return (
    <div className="cua-stats">
      {stats.duration.toFixed(1)}s · {stats.iterations} 步 · {totalTokens}{" "}
      tokens
    </div>
  );
};
