import React, { useEffect, useState } from "react";
import type { ExperienceNotice } from "../types";

interface Props {
  notice: ExperienceNotice;
}

export const ExperienceNoticeBar: React.FC<Props> = ({ notice }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Trigger entrance animation on next frame
    requestAnimationFrame(() => setVisible(true));
  }, []);

  return (
    <div className={`cua-experience ${visible ? "cua-experience-visible" : ""}`}>
      <div className="cua-experience-glow" />
      <span className="cua-experience-icon">🧠</span>
      <div className="cua-experience-content">
        <span className="cua-experience-label">经验升级</span>
        <span className="cua-experience-title">{notice.title}</span>
      </div>
    </div>
  );
};
