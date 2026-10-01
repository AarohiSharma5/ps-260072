import type { Scenario } from "./types"

export function formatNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Unavailable"
  return value.toFixed(digits)
}

export function formatSigned(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Unavailable"
  const text = value.toFixed(digits)
  return value > 0 ? `+${text}` : text
}

export function coordinates(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S"
  const ew = lon >= 0 ? "E" : "W"
  return `${Math.abs(lat).toFixed(3)}° ${ns}, ${Math.abs(lon).toFixed(3)}° ${ew}`
}

export function riskLevel(probability: number | null, thresholds: Scenario["thresholds"]): string {
  if (probability === null || Number.isNaN(probability)) return "UNAVAILABLE"
  if (probability >= thresholds.severe) return "SEVERE RISK"
  if (probability >= thresholds.high) return "HIGH RISK"
  if (probability >= thresholds.watch) return "WATCH"
  return "NORMAL"
}

export function riskClass(level: string): string {
  if (level === "SEVERE RISK") return "risk-severe"
  if (level === "HIGH RISK") return "risk-high"
  if (level === "WATCH") return "risk-watch"
  if (level === "NORMAL") return "risk-normal"
  return "risk-unknown"
}

export function confidenceMargin(probabilities: number[]): number | null {
  if (probabilities.length === 0 || probabilities.some((value) => Number.isNaN(value))) return null
  const margin = probabilities.reduce((sum, value) => sum + Math.abs(value - 0.5) * 2, 0) / probabilities.length
  return margin
}

export function metricText(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Undefined"
  return value.toFixed(digits)
}

export function clockStamp(iso: string | null | undefined): string {
  if (!iso) return "Unavailable"
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "short",
    timeZone: "UTC",
    hourCycle: "h23",
  }).format(date) + " UTC"
}

export function maxOf(values: (number | null)[] | null | undefined): number | null {
  const clean = (values ?? []).filter((value): value is number => value !== null)
  if (clean.length === 0) return null
  return Math.max(...clean)
}

export function meanOf(values: (number | null)[] | null | undefined): number | null {
  const clean = (values ?? []).filter((value): value is number => value !== null)
  if (clean.length === 0) return null
  return clean.reduce((sum, value) => sum + value, 0) / clean.length
}
