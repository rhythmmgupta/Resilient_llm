import type { ChatResponse, HealthResponse, SessionHistoryResponse } from "./types";

const DEFAULT_API_BASE_URL =
  typeof window !== "undefined" &&
  (window.location.hostname === "resilientllmfrontend.vercel.app" ||
    window.location.hostname.endsWith("-rhythm-0db7.vercel.app"))
    ? "https://resilient-llm.onrender.com"
    : "http://localhost:8000";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function formatErrorDetail(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const loc = Array.isArray((item as { loc?: unknown[] }).loc)
            ? (item as { loc: unknown[] }).loc.join(".")
            : "";
          return loc ? `${loc}: ${String((item as { msg: unknown }).msg)}` : String((item as { msg: unknown }).msg);
        }
        return JSON.stringify(item);
      })
      .join("; ");
  }
  return fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, formatErrorDetail(data, `Health check failed (${response.status})`));
  }
  return data as HealthResponse;
}

export async function sendChatMessage(params: {
  sessionId: string;
  message: string;
  systemPrompt?: string;
  simulateFailure?: string;
}): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (params.simulateFailure && params.simulateFailure.trim()) {
    headers["X-Simulate-Failure"] = params.simulateFailure.trim();
  }

  const body: Record<string, string> = {
    session_id: params.sessionId,
    message: params.message,
  };
  if (params.systemPrompt && params.systemPrompt.trim()) {
    body.system_prompt = params.systemPrompt.trim();
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, formatErrorDetail(data, `Request failed with status ${response.status}`));
  }

  return data as ChatResponse;
}

export async function fetchSessionMessages(sessionId: string): Promise<SessionHistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/messages`);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, formatErrorDetail(data, `Failed to load session (${response.status})`));
  }
  return data as SessionHistoryResponse;
}

export { API_BASE_URL };
