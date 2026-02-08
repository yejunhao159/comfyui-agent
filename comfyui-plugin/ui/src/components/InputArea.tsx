import React, { useCallback, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onCancel: () => void;
  isProcessing: boolean;
  disabled: boolean;
}

export const InputArea: React.FC<Props> = ({
  onSend,
  onCancel,
  isProcessing,
  disabled,
}) => {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [text, disabled, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-resize
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <div className="cua-input-area">
      <textarea
        ref={textareaRef}
        className="cua-input"
        value={text}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="输入消息..."
        rows={1}
        disabled={disabled}
      />
      {isProcessing ? (
        <button className="cua-btn cua-btn-cancel" onClick={onCancel}>
          停止
        </button>
      ) : (
        <button
          className="cua-btn cua-btn-send"
          onClick={handleSend}
          disabled={disabled || !text.trim()}
        >
          发送
        </button>
      )}
    </div>
  );
};
