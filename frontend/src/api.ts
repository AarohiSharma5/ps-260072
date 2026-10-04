import type { LiveForecast, NowcastCity, NowcastDay, NowcastDayInfo, NowcastDomain, NowcastSummary } from "./types"

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
  health: () => getJson<{ ok: boolean; model_loaded: boolean; model_version: string; nowcast_cities?: string[]; live?: boolean }>("/api/health"),
  nowcastCities: () => getJson<{ default: string; cities: NowcastCity[]; live: boolean }>("/api/nowcast/cities"),
  nowcastDomains: () => getJson<{ domains: NowcastDomain[] }>("/api/nowcast/domains"),
  nowcastSummary: (city: string) => getJson<NowcastSummary>(`/api/nowcast/${city}/summary`),
  nowcastDays: (city: string) => getJson<{ days: NowcastDayInfo[] }>(`/api/nowcast/${city}/days`),
  nowcastDay: (city: string, day: string) => getJson<NowcastDay>(`/api/nowcast/${city}/day?day=${encodeURIComponent(day)}`),
  nowcastLive: (city: string) => getJson<LiveForecast>(`/api/nowcast/${city}/live`),
}
