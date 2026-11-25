export type ResolutionSource = "LIVE_INFERENCE" | "LLM_CACHE";

export interface ChatMetadata {
  resolution_source: ResolutionSource;
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ChatMetadata;
}

export interface HealthResponse {
  status: string;
  keys_configured: number;
  cache_enabled: boolean;
}

export interface SessionHistoryMessage {
  role: "user" | "assistant";
  content: string;
  metadata?: Partial<ChatMetadata>;
}

export interface SessionHistoryResponse {
  session_id: string;
  messages: SessionHistoryMessage[];
}

export type ChatEntry =
  | {
      id: string;
      kind: "user";
      content: string;
    }
  | {
      id: string;
      kind: "assistant";
      content: string;
      metadata: ChatMetadata;
    }
  | {
      id: string;
      kind: "error";
      statusCode?: number;
      detail: string;
      failedMessage?: string;
    };
