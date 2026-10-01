import type { Explanation, Methodology, MetricsReport, Scenario, Status } from "./types"

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* response was not JSON */
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export const api = {
  health: () => getJson<{ ok: boolean; model_loaded: boolean; model_version: string }>("/api/health"),
  status: () => getJson<Status>("/api/status"),
  cases: () => getJson<{ data_mode: string; cases: Scenario["case"][] }>("/api/cases"),
  scenario: (caseId: string) => getJson<Scenario>(`/api/scenario?case_id=${encodeURIComponent(caseId)}`),
  metrics: () => getJson<MetricsReport>("/api/metrics"),
  methodology: () => getJson<Methodology>("/api/methodology"),
  explanation: (caseId: string, gridId: number, hazard: string, lead: number) =>
    getJson<Explanation>(
      `/api/cases/${encodeURIComponent(caseId)}/cells/${gridId}/explanation?hazard=${hazard}&lead=${lead}`,
    ),
}
