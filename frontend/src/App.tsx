import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import {
  API_BASE_URL,
  ApiError,
  fetchHealth,
  fetchSessionMessages,
  sendChatMessage,
} from "./api";
import type { ChatEntry, HealthResponse } from "./types";

function generateSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

function nextEntryId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const SIMULATION_PRESETS = [
  { label: "Normal (No Failure)", value: "" },
  { label: "key_1=429 (Rate Limit)", value: "key_1=429" },
  { label: "key_1=500 (Server Error)", value: "key_1=500" },
  { label: "*=429 (All Keys Down)", value: "*=429" },
];

const QUICK_PROMPTS = [
  "What is the capital of France?",
  "Remember my favorite language is Python.",
  "What is my favorite language?",
];

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [input, setInput] = useState<string>("");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [simulateFailure, setSimulateFailure] = useState<string>("");
  const [showLab, setShowLab] = useState<boolean>(false);
  const [sessionLookupId, setSessionLookupId] = useState<string>("");

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const loadHealth = async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : "Backend unreachable");
    }
  };

  useEffect(() => {
    void loadHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, isSending]);

  const handleNewSession = () => {
    setSessionId(generateSessionId());
    setEntries([]);
    setInput("");
  };

  const handleLoadSessionFromDb = async () => {
    const targetId = (sessionLookupId.trim() || sessionId).trim();
    if (!targetId) return;
    try {
      const data = await fetchSessionMessages(targetId);
      setSessionId(data.session_id);
      const loadedEntries: ChatEntry[] = data.messages.map((msg) => {
        if (msg.role === "user") {
          return {
            id: nextEntryId(),
            kind: "user",
            content: msg.content,
          };
        }
        return {
          id: nextEntryId(),
          kind: "assistant",
          content: msg.content,
          metadata: {
            resolution_source: msg.metadata?.resolution_source ?? "LIVE_INFERENCE",
            model: msg.metadata?.model ?? "unknown",
            latency_ms: msg.metadata?.latency_ms ?? 0,
            input_tokens: msg.metadata?.input_tokens ?? 0,
            output_tokens: msg.metadata?.output_tokens ?? 0,
            key_id: msg.metadata?.key_id ?? null,
            fallback_attempts: msg.metadata?.fallback_attempts ?? 0,
          },
        };
      });
      setEntries(loadedEntries);
      setSessionLookupId("");
    } catch (err) {
      const statusCode = err instanceof ApiError ? err.status : undefined;
      const detail =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Failed to load session";
      setEntries((prev) => [
        ...prev,
        {
          id: nextEntryId(),
          kind: "error",
          statusCode,
          detail,
        },
      ]);
    }
  };

  const submitMessage = async (overrideMessage?: string) => {
    if (isSending) return;

    const outgoingMessage = overrideMessage !== undefined ? overrideMessage : input;
    if (overrideMessage === undefined) {
      setInput("");
    }
    setIsSending(true);

    try {
      const data = await sendChatMessage({
        sessionId,
        message: outgoingMessage,
        systemPrompt,
        simulateFailure,
      });

      setEntries((prev) => [
        ...prev,
        {
          id: nextEntryId(),
          kind: "user",
          content: outgoingMessage.trim(),
        },
        {
          id: nextEntryId(),
          kind: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch (err) {
      const statusCode = err instanceof ApiError ? err.status : undefined;
      const detail =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Unable to reach the backend service";

      setEntries((prev) => [
        ...prev,
        {
          id: nextEntryId(),
          kind: "error",
          statusCode,
          detail,
          failedMessage: outgoingMessage.trim() || undefined,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitMessage();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submitMessage();
    }
  };

  const trimmedLength = input.trim().length;

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand-group">
          <div className="brand-mark" aria-hidden="true">
            LG
          </div>
          <div>
            <h1 className="brand-title">Resilient LLM Gateway</h1>
            <p className="brand-subtitle">
              LiteLLM Key Failover · Response Caching · SQLite Multi-Turn State
            </p>
          </div>
        </div>

        <div className="header-actions">
          <button
            type="button"
            className={`health-pill ${health ? "health-ok" : "health-down"}`}
            title={`Backend: ${API_BASE_URL} (click to refresh)`}
            onClick={() => void loadHealth()}
          >
            <span className="health-dot" />
            {health ? (
              <span>
                {health.status} · {health.keys_configured} keys ·{" "}
                {health.cache_enabled ? "cache on" : "cache off"}
              </span>
            ) : (
              <span>{healthError || "checking..."}</span>
            )}
          </button>

          <span className="session-badge" title="Current active session ID">
            Session: <code>{sessionId}</code>
          </span>

          <button type="button" className="btn-secondary" onClick={handleNewSession}>
            New Session
          </button>

          <button
            type="button"
            className={`btn-secondary ${showLab ? "btn-active" : ""}`}
            onClick={() => setShowLab((v) => !v)}
          >
            Failover Lab
          </button>
        </div>
      </header>

      <section className="config-strip" aria-label="Session configuration">
        <label htmlFor="system-prompt-input" className="config-label">
          System Prompt (optional)
        </label>
        <input
          id="system-prompt-input"
          data-testid="system-prompt-input"
          type="text"
          className="system-prompt-field"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="Override default system prompt for this session (e.g., Answer in one concise sentence)..."
        />
      </section>

      {showLab && (
        <section className="lab-panel" aria-label="Failover and session inspector">
          <div className="lab-row">
            <div className="lab-group">
              <span className="lab-heading">Simulate Failure Header (X-Simulate-Failure)</span>
              <div className="preset-pills">
                {SIMULATION_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    className={`preset-btn ${simulateFailure === preset.value ? "preset-active" : ""}`}
                    onClick={() => setSimulateFailure(preset.value)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <input
                type="text"
                className="lab-input"
                value={simulateFailure}
                onChange={(e) => setSimulateFailure(e.target.value)}
                placeholder="Custom e.g. key_1=429,key_2=500"
                aria-label="Custom X-Simulate-Failure header"
              />
            </div>

            <div className="lab-group">
              <span className="lab-heading">Inspect SQLite Session History</span>
              <div className="lookup-row">
                <input
                  type="text"
                  className="lab-input"
                  value={sessionLookupId}
                  onChange={(e) => setSessionLookupId(e.target.value)}
                  placeholder={`Session ID (default: ${sessionId})`}
                  aria-label="Session ID to load from SQLite"
                />
                <button type="button" className="btn-secondary" onClick={() => void handleLoadSessionFromDb()}>
                  Load from DB
                </button>
              </div>
              <span className="lab-hint">
                Verify: repeat a first prompt after <strong>New Session</strong> for{" "}
                <code>LLM_CACHE</code>, or select <code>key_1=429</code> for automatic{" "}
                <code>key_2</code> fallback.
              </span>
            </div>
          </div>
        </section>
      )}

      <main className="chat-container">
        {entries.length === 0 && !isSending ? (
          <div className="empty-state">
            <h2>Start a multi-turn conversation</h2>
            <p>
              Messages are stored per session in SQLite. Repeat an identical initial prompt across a{" "}
              <strong>New Session</strong> to observe <code>LLM_CACHE</code> speedup, or open{" "}
              <strong>Failover Lab</strong> to test <code>X-Simulate-Failure</code> key fallback.
            </p>
            <div className="quick-prompts">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="quick-prompt-btn"
                  onClick={() => void submitMessage(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {entries.map((entry) => {
              if (entry.kind === "user") {
                return (
                  <div key={entry.id} className="message-row user-row">
                    <div className="user-bubble">
                      <div className="role-tag">You</div>
                      <div className="message-text">{entry.content}</div>
                    </div>
                  </div>
                );
              }

              if (entry.kind === "error") {
                return (
                  <div key={entry.id} className="message-row error-row" role="alert">
                    <div className="error-bubble">
                      <div className="error-header">
                        <span className="error-title">
                          Request Failed{entry.statusCode ? ` (HTTP ${entry.statusCode})` : ""}
                        </span>
                        {entry.failedMessage && (
                          <button
                            type="button"
                            className="retry-btn"
                            disabled={isSending}
                            onClick={() => void submitMessage(entry.failedMessage)}
                          >
                            Retry
                          </button>
                        )}
                      </div>
                      <div className="error-detail">{entry.detail}</div>
                    </div>
                  </div>
                );
              }

              const isCache = entry.metadata.resolution_source === "LLM_CACHE";

              return (
                <div key={entry.id} className="message-row assistant-row">
                  <div className="assistant-bubble" data-testid="assistant-message">
                    <div className="role-tag">Assistant</div>
                    <div className="message-text">{entry.content}</div>

                    <div className="meta-bar">
                      <span
                        data-testid="meta-source"
                        className={`meta-badge ${isCache ? "badge-cache" : "badge-live"}`}
                      >
                        {entry.metadata.resolution_source}
                      </span>

                      <span className="meta-chip">
                        Model: <strong data-testid="meta-model">{entry.metadata.model}</strong>
                      </span>

                      <span className="meta-chip">
                        Latency: <strong data-testid="meta-latency">{entry.metadata.latency_ms} ms</strong>
                      </span>

                      <span className="meta-chip">
                        In:{" "}
                        <strong data-testid="meta-input-tokens">{entry.metadata.input_tokens}</strong>
                      </span>

                      <span className="meta-chip">
                        Out:{" "}
                        <strong data-testid="meta-output-tokens">{entry.metadata.output_tokens}</strong>
                      </span>

                      <span className="meta-chip">
                        Key: <strong>{entry.metadata.key_id ?? "null"}</strong>
                      </span>

                      {entry.metadata.fallback_attempts > 0 && (
                        <span className="meta-badge badge-failover">
                          failover ×{entry.metadata.fallback_attempts}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}

            {isSending && (
              <div className="message-row assistant-row">
                <div className="assistant-bubble typing-bubble">
                  <span className="typing-dots">Generating response...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="composer-bar">
        {simulateFailure && (
          <div className="sim-banner">
            Active header: <code>X-Simulate-Failure: {simulateFailure}</code>
          </div>
        )}
        <form className="composer-form" onSubmit={handleSubmit}>
          <textarea
            data-testid="chat-input"
            className="chat-textarea"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message... (Enter to send, Shift+Enter for new line)"
          />
          <button
            data-testid="send-button"
            type="submit"
            className="send-btn"
            disabled={isSending}
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </form>
        <div className="composer-meta">
          <span className={trimmedLength > 4000 ? "char-over" : ""}>
            {trimmedLength} / 4000 chars
          </span>
        </div>
      </footer>
    </div>
  );
}
