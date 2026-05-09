// services/api.ts
// All backend calls live here.
// Your PC's Wi-Fi IP is hardcoded below — phone and PC must be on same Wi-Fi.

export const BASE_URL = "http://192.168.10.6:5000";

export interface AnalyzeResponse {
  success: boolean;
  wound_type: string;
  wound_label: string;
  confidence: number;
  confidence_percent: string;
  severity_level: string;
  seek_emergency: boolean;
  first_aid: {
    steps: string[];
    do_not: string[];
  };
}

// ── ANALYZE ───────────────────────────────────────────────────────────────────
/** Sends image only; backend defaults severity/bleeding/swelling for AI-focused flow. */
export async function analyzeWound(imageUri: string): Promise<AnalyzeResponse> {
  const formData = new FormData();

  const filename = imageUri.split("/").pop() ?? "wound.jpg";
  const ext = filename.split(".").pop()?.toLowerCase() ?? "jpg";
  const mimeType = ext === "png" ? "image/png" : "image/jpeg";

  formData.append("image", {
    uri: imageUri,
    name: filename,
    type: mimeType,
  } as any);

  formData.append("severity", "mild");
  formData.append("bleeding", "false");
  formData.append("swelling", "false");

  const response = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error?.error ?? `Server error: ${response.status}`);
  }

  return response.json();
}

// ── HEALTH CHECK ──────────────────────────────────────────────────────────────
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/api/health`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}
