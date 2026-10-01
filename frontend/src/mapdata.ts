import type { BaseLayer, Hazard, Overlays, Scenario, Series } from "./types"

export interface Stop {
  at: number
  color: string
  fade: number
}

export interface LegendModel {
  title: string
  unit: string
  stops: Stop[]
  caption: string
  minLabel: string
  maxLabel: string
}

interface Feature {
  type: "Feature"
  properties: Record<string, string | number>
  geometry: { type: string; coordinates: unknown }
}

interface Collection {
  type: "FeatureCollection"
  features: Feature[]
}

const PROBABILITY: Stop[] = [
  { at: 0, color: "#163044", fade: 0 },
  { at: 0.2, color: "#1f6f8a", fade: 0.38 },
  { at: 0.35, color: "#2a9d8f", fade: 0.58 },
  { at: 0.6, color: "#e0b15a", fade: 0.74 },
  { at: 0.8, color: "#ef7a3a", fade: 0.84 },
  { at: 1, color: "#d6453d", fade: 0.92 },
]

const SCALES: Record<string, { title: string; unit: string; stops: Stop[]; caption: string }> = {
  vil: {
    title: "Vertically integrated liquid",
    unit: "kg/m²",
    caption: "SEVIR radar VIL, averaged to 8 km. The thunderstorm proxy is 5 kg/m². Not dBZ.",
    stops: [
      { at: 0, color: "#101418", fade: 0 },
      { at: 1, color: "#3d5f8a", fade: 0.28 },
      { at: 3, color: "#3e8f86", fade: 0.48 },
      { at: 5, color: "#3f8f45", fade: 0.62 },
      { at: 10, color: "#d2c45a", fade: 0.74 },
      { at: 20, color: "#e08a32", fade: 0.84 },
      { at: 40, color: "#d24b3a", fade: 0.92 },
    ],
  },
  reflectivity: {
    title: "Radar reflectivity",
    unit: "dBZ",
    caption: "Reflectivity at the analysis time.",
    stops: [
      { at: 0, color: "#101418", fade: 0 },
      { at: 8, color: "#3d5f8a", fade: 0.28 },
      { at: 18, color: "#3e8f86", fade: 0.48 },
      { at: 28, color: "#3f8f45", fade: 0.62 },
      { at: 38, color: "#d2c45a", fade: 0.74 },
      { at: 48, color: "#e08a32", fade: 0.84 },
      { at: 58, color: "#d24b3a", fade: 0.9 },
      { at: 68, color: "#f4f1ea", fade: 0.95 },
    ],
  },
  ir_bt: {
    title: "Brightness temperature",
    unit: "K",
    caption: "Colder tops are brighter.",
    stops: [
      { at: 190, color: "#fffaf0", fade: 0.9 },
      { at: 220, color: "#f0e2a8", fade: 0.8 },
      { at: 250, color: "#c9c3b4", fade: 0.55 },
      { at: 280, color: "#35586e", fade: 0.35 },
      { at: 310, color: "#1a232b", fade: 0.12 },
    ],
  },
  lightning_density: {
    title: "Lightning density",
    unit: "flashes / km² / step",
    caption: "Flash density on the analysis grid.",
    stops: [
      { at: 0, color: "#102028", fade: 0 },
      { at: 0.02, color: "#1a6d86", fade: 0.4 },
      { at: 0.08, color: "#7fd3ea", fade: 0.7 },
      { at: 0.2, color: "#f4f7fb", fade: 0.9 },
    ],
  },
  convective_precip: {
    title: "Convective precipitation",
    unit: "mm/h",
    caption: "ERA5 convective precipitation. The thunderstorm proxy label is this field at or above 1 mm/h.",
    stops: [
      { at: 0, color: "#101418", fade: 0 },
      { at: 0.2, color: "#2d5f7a", fade: 0.35 },
      { at: 1, color: "#3f8f45", fade: 0.6 },
      { at: 3, color: "#d2c45a", fade: 0.75 },
      { at: 6, color: "#e08a32", fade: 0.85 },
      { at: 12, color: "#d24b3a", fade: 0.92 },
    ],
  },
  temperature: {
    title: "Temperature",
    unit: "°C",
    caption: "2 m temperature.",
    stops: [
      { at: 18, color: "#234863", fade: 0.45 },
      { at: 30, color: "#d7a441", fade: 0.55 },
      { at: 42, color: "#d4543c", fade: 0.7 },
    ],
  },
  rh: {
    title: "Relative humidity",
    unit: "%",
    caption: "Relative humidity at 2 m.",
    stops: [
      { at: 10, color: "#3a342c", fade: 0.35 },
      { at: 55, color: "#3f6f62", fade: 0.55 },
      { at: 100, color: "#d5ebe4", fade: 0.75 },
    ],
  },
  cape: {
    title: "CAPE",
    unit: "J/kg",
    caption: "Convective available potential energy.",
    stops: [
      { at: 0, color: "#1c242c", fade: 0.15 },
      { at: 500, color: "#2a5270", fade: 0.4 },
      { at: 1500, color: "#e0b15a", fade: 0.65 },
      { at: 3000, color: "#d4543c", fade: 0.8 },
    ],
  },
  cin: {
    title: "CIN",
    unit: "J/kg",
    caption: "Negative values are inhibition.",
    stops: [
      { at: -350, color: "#1d3348", fade: 0.72 },
      { at: -150, color: "#5d7a92", fade: 0.55 },
      { at: -20, color: "#e8d7b0", fade: 0.5 },
    ],
  },
  shear: {
    title: "Bulk shear",
    unit: "m/s",
    caption: "Shear magnitude.",
    stops: [
      { at: 0, color: "#1c242c", fade: 0.2 },
      { at: 10, color: "#3d6b8a", fade: 0.5 },
      { at: 20, color: "#d7c4a3", fade: 0.7 },
      { at: 30, color: "#f2f2f2", fade: 0.8 },
    ],
  },
  k_index: {
    title: "K index",
    unit: "°C",
    caption: "Moisture and lapse-rate index. Above about 30 favours thunderstorms.",
    stops: [
      { at: 0, color: "#1c242c", fade: 0.15 },
      { at: 20, color: "#2a5270", fade: 0.4 },
      { at: 30, color: "#e0b15a", fade: 0.65 },
      { at: 40, color: "#d4543c", fade: 0.8 },
    ],
  },
  total_totals: {
    title: "Totals-Totals",
    unit: "°C",
    caption: "Stability index. Above about 50 favours storms.",
    stops: [
      { at: 30, color: "#1c242c", fade: 0.15 },
      { at: 44, color: "#2a5270", fade: 0.4 },
      { at: 50, color: "#e0b15a", fade: 0.65 },
      { at: 58, color: "#d4543c", fade: 0.8 },
    ],
  },
}

function mix(a: string, b: string, t: number): string {
  const parse = (hex: string) => [0, 2, 4].map((index) => parseInt(hex.slice(1 + index, 3 + index), 16))
  const left = parse(a)
  const right = parse(b)
  const channels = left.map((channel, index) => Math.round(channel + (right[index] - channel) * t))
  return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`
}

export function sampleStop(stops: Stop[], value: number): { color: string; fade: number } {
  if (value <= stops[0].at) return { color: stops[0].color, fade: stops[0].fade }
  const last = stops[stops.length - 1]
  if (value >= last.at) return { color: last.color, fade: last.fade }
  for (let index = 1; index < stops.length; index += 1) {
    const next = stops[index]
    const prev = stops[index - 1]
    if (value <= next.at) {
      const span = next.at - prev.at || 1
      const t = (value - prev.at) / span
      return { color: mix(prev.color, next.color, t), fade: prev.fade + (next.fade - prev.fade) * t }
    }
  }
  return { color: last.color, fade: last.fade }
}

export function rampGradient(stops: Stop[]): string {
  const lo = stops[0].at
  const hi = stops[stops.length - 1].at
  const span = hi - lo || 1
  return `linear-gradient(90deg, ${stops.map((stop) => `${stop.color} ${((stop.at - lo) / span) * 100}%`).join(", ")})`
}

function empty(): Collection {
  return { type: "FeatureCollection", features: [] }
}

function verifyIndex(scenario: Scenario, offset: number): number {
  return scenario.verification.offsets_min.indexOf(offset)
}

export function isForecastBase(base: BaseLayer): boolean {
  return base === "thunderstorm" || base === "lightning" || base === "extrapolation"
}

export function hazardForBase(base: BaseLayer, hazard: Hazard, scenario: Scenario): Hazard {
  const wanted: Hazard = base === "lightning" ? "lightning" : base === "extrapolation" ? hazard : "thunderstorm"
  return scenario.hazards.includes(wanted) ? wanted : scenario.hazards[0]
}

export function activeModelLead(offset: number, scenario: Scenario): number {
  return offset === 0 ? scenario.panel_leads_min[0] ?? scenario.leads_min[0] : offset
}

export function legendFor(scenario: Scenario, base: BaseLayer, hazard: Hazard, offset: number, overlays: Overlays): LegendModel {
  if (isForecastBase(base)) {
    const h = hazardForBase(base, hazard, scenario)
    const name = h === "lightning" ? "Lightning" : "Thunderstorm"
    const kind = base === "extrapolation" ? "Motion extrapolation" : "Model probability"
    const labelNote = scenario.data_mode === "HISTORICAL" && h === "thunderstorm" ? " Proxy label: ERA5 convective precipitation ≥ 1 mm/h." : ""
    return {
      title: offset === 0 ? `${name} analysis indicator` : `${name} · ${kind}`,
      unit: offset === 0 ? "event / no event" : "probability",
      stops: PROBABILITY,
      minLabel: "0",
      maxLabel: "1",
      caption:
        offset === 0
          ? `Analysis indicator: the ${h} label at the analysis time. Not a model probability.${labelNote}`
          : base === "extrapolation"
            ? `Current event mask advected ${offset} min by estimated reflectivity motion. Baseline, not the XGBoost probability.`
            : `XGBoost probability of the prototype ${h} label at +${offset} min.${labelNote}`,
    }
  }
  const scale = SCALES[base]
  const showingFuture = overlays.verify && offset > 0 && (base === "reflectivity" || base === "vil" || base === "lightning_density" || base === "convective_precip")
  const unit = base === "lightning_density" ? `flashes / km² / ${scenario.step_text}` : scale.unit
  return {
    title: scale.title,
    unit,
    stops: scale.stops,
    minLabel: String(scale.stops[0].at),
    maxLabel: String(scale.stops[scale.stops.length - 1].at),
    caption: showingFuture
      ? `${scenario.domain.name}: dataset state at this valid time. Verification is on. This is not a forecast.`
      : `${scenario.domain.name}. ${scale.caption} ${scenario.data_mode}.`,
  }
}

function fieldValues(scenario: Scenario, base: BaseLayer, hazard: Hazard, offset: number, overlays: Overlays): Series | null {
  const a = scenario.analysis
  if (base === "thunderstorm" || base === "lightning") {
    const h = hazardForBase(base, hazard, scenario)
    if (offset === 0) return h === "thunderstorm" ? a.thunderstorm_indicator : a.lightning_indicator
    return scenario.prediction[h]?.[String(offset)] ?? null
  }
  if (base === "extrapolation") {
    const h = hazardForBase(base, hazard, scenario)
    if (offset === 0) return scenario.persistence[h] ?? null
    return scenario.extrapolation?.[h]?.[String(offset)] ?? null
  }
  const vi = verifyIndex(scenario, offset)
  if (overlays.verify && offset > 0 && vi >= 0) {
    if (base === "reflectivity") return scenario.verification.reflectivity?.[vi] ?? null
    if (base === "vil") return scenario.verification.vil?.[vi] ?? null
    if (base === "convective_precip") return scenario.verification.convective_precip?.[vi] ?? null
    if (base === "lightning_density") {
      const counts = scenario.verification.lightning_count?.[vi]
      if (!counts) return null
      const area = scenario.domain.cell_km_lat * scenario.domain.cell_km_lon
      return counts.map((count) => (count === null ? null : count / area))
    }
  }
  const direct: Partial<Record<BaseLayer, Series | null>> = {
    reflectivity: a.reflectivity,
    vil: a.vil,
    ir_bt: a.ir_bt,
    lightning_density: a.lightning_density,
    convective_precip: a.convective_precip,
    temperature: a.temperature,
    rh: a.rh,
    cape: a.cape,
    cin: a.cin,
    shear: a.shear,
    k_index: a.k_index,
    total_totals: a.total_totals,
  }
  return direct[base] ?? null
}

function stopsFor(base: BaseLayer): Stop[] {
  return isForecastBase(base) ? PROBABILITY : SCALES[base].stops
}

export function buildCollections(scenario: Scenario, base: BaseLayer, hazard: Hazard, offset: number, overlays: Overlays) {
  const values = fieldValues(scenario, base, hazard, offset, overlays)
  const stops = stopsFor(base)
  const verifyAt = verifyIndex(scenario, offset)
  const stormVerify = overlays.verify && verifyAt >= 0 ? scenario.verification.thunderstorm?.[verifyAt] : null
  const cells: Collection = empty()
  scenario.grid.forEach((cell, index) => {
    const value = values ? values[index] : null
    const painted = value === null || value === undefined ? { color: "#000000", fade: 0 } : sampleStop(stops, value)
    cells.features.push({
      type: "Feature",
      properties: {
        id: cell.grid_id,
        value: value === null || value === undefined ? "" : Number(value.toFixed(3)),
        color: painted.color,
        fade: value === null || value === undefined ? 0 : painted.fade,
        verify: stormVerify && (stormVerify[index] ?? 0) >= 1 ? 1 : 0,
      },
      geometry: { type: "Polygon", coordinates: [cell.polygon] },
    })
  })

  const wind: Collection = empty()
  if (overlays.wind && scenario.analysis.wind_u && scenario.analysis.wind_v) {
    const stride = Math.max(1, Math.round(Math.max(scenario.domain.n_lat, scenario.domain.n_lon) / 12))
    scenario.grid.forEach((cell, index) => {
      if (cell.row % stride !== 0 || cell.col % stride !== 0) return
      const u = scenario.analysis.wind_u?.[index]
      const v = scenario.analysis.wind_v?.[index]
      if (u === undefined || v === undefined || u === null || v === null) return
      const speed = Math.hypot(u, v)
      if (speed < 0.5) return
      const km = Math.min(speed, 22) * 0.06 * scenario.domain.cell_km_lat
      const dLat = ((v / speed) * km) / 110.54
      const dLon = ((u / speed) * km) / (111.32 * Math.cos((cell.latitude * Math.PI) / 180))
      wind.features.push({
        type: "Feature",
        properties: { speed: Number(speed.toFixed(1)) },
        geometry: { type: "LineString", coordinates: [[cell.longitude, cell.latitude], [cell.longitude + dLon, cell.latitude + dLat]] },
      })
    })
  }

  const motion: Collection = empty()
  const tracked = scenario.analysis.reflectivity ?? scenario.analysis.vil
  const trackCut = scenario.analysis.reflectivity ? 20 : 5
  if (overlays.motion && scenario.analysis.motion_u && scenario.analysis.motion_v && tracked) {
    const seconds = Math.max(offset, scenario.step_min) * 60
    const ranked = scenario.grid
      .map((cell, index) => ({ cell, index, refl: tracked[index] ?? 0 }))
      .filter((item) => (item.refl ?? 0) >= trackCut)
      .sort((a, b) => (b.refl ?? 0) - (a.refl ?? 0))
    const picked = ranked.filter((item) => isLocalMax(scenario, item.index)).slice(0, 16)
    const arrows = picked.length > 0 ? picked : ranked.slice(0, 8)
    arrows.forEach(({ cell, index }) => {
      const u = scenario.analysis.motion_u?.[index] ?? 0
      const v = scenario.analysis.motion_v?.[index] ?? 0
      if (u === null || v === null) return
      const speed = Math.hypot(u, v)
      if (speed < 1) return
      const dLat = (v * seconds) / 110540
      const dLon = (u * seconds) / (111320 * Math.cos((cell.latitude * Math.PI) / 180))
      motion.features.push({
        type: "Feature",
        properties: { speed: Number((speed * 3.6).toFixed(1)) },
        geometry: { type: "LineString", coordinates: [[cell.longitude, cell.latitude], [cell.longitude + dLon, cell.latitude + dLat]] },
      })
    })
  }

  const lightning: Collection = empty()
  if (overlays.lightning && scenario.hazards.includes("lightning")) {
    if (offset === 0 && scenario.analysis.lightning_count) {
      scenario.grid.forEach((cell, index) => {
        const count = scenario.analysis.lightning_count?.[index] ?? 0
        if (count === null || count < 1) return
        lightning.features.push(point(cell.longitude, cell.latitude, Math.min(4 + Math.sqrt(count) * 2.2, 14), 0.9, "analysis"))
      })
    } else if (offset > 0) {
      const probs = scenario.prediction.lightning?.[String(offset)]
      if (probs) {
        scenario.grid.forEach((cell, index) => {
          const probability = probs[index]
          if (probability === null || probability < 0.35) return
          lightning.features.push(point(cell.longitude, cell.latitude, 4 + probability * 7, 0.35 + probability * 0.5, "model"))
        })
      }
    }
  }

  const verifyFlashes: Collection = empty()
  if (overlays.verify && offset > 0 && verifyAt >= 0 && scenario.verification.lightning_count) {
    const counts = scenario.verification.lightning_count[verifyAt]
    if (counts) {
      scenario.grid.forEach((cell, index) => {
        if ((counts[index] ?? 0) < 1) return
        verifyFlashes.features.push(point(cell.longitude, cell.latitude, 6, 1, "verify"))
      })
    }
  }

  return { cells, wind, motion, lightning, verifyFlashes, graticule: buildGraticule(scenario), domain: buildDomain(scenario) }
}

function point(lon: number, lat: number, radius: number, fade: number, kind: string): Feature {
  return { type: "Feature", properties: { radius: Number(radius.toFixed(2)), fade, kind }, geometry: { type: "Point", coordinates: [lon, lat] } }
}

function isLocalMax(scenario: Scenario, index: number): boolean {
  const refl = scenario.analysis.reflectivity
  if (!refl) return false
  const cell = scenario.grid[index]
  const value = refl[index] ?? 0
  const nlon = scenario.domain.n_lon
  for (let dr = -1; dr <= 1; dr += 1) {
    for (let dc = -1; dc <= 1; dc += 1) {
      if (dr === 0 && dc === 0) continue
      const row = cell.row + dr
      const col = cell.col + dc
      if (row < 0 || col < 0 || row >= scenario.domain.n_lat || col >= scenario.domain.n_lon) continue
      if ((refl[row * nlon + col] ?? 0) > value) return false
    }
  }
  return value >= 28
}

function buildGraticule(scenario: Scenario): Collection {
  const { lat_min, lat_max, lon_min, lon_max } = scenario.domain
  const step = lat_max - lat_min > 4 ? 1 : 0.5
  const features: Feature[] = []
  for (let lat = Math.ceil(lat_min / step) * step; lat < lat_max; lat += step) features.push(line([[lon_min, lat], [lon_max, lat]]))
  for (let lon = Math.ceil(lon_min / step) * step; lon < lon_max; lon += step) features.push(line([[lon, lat_min], [lon, lat_max]]))
  return { type: "FeatureCollection", features }
}

function buildDomain(scenario: Scenario): Collection {
  const { lat_min, lat_max, lon_min, lon_max } = scenario.domain
  return { type: "FeatureCollection", features: [line([[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]])] }
}

function line(coordinates: number[][]): Feature {
  return { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates } }
}

export function combinedProbability(scenario: Scenario, lead: number): Series | null {
  const series = scenario.hazards.map((h) => scenario.prediction[h]?.[String(lead)]).filter((s): s is Series => !!s)
  if (series.length === 0) return null
  return series[0].map((_, index) => {
    const values = series.map((s) => s[index]).filter((v): v is number => v !== null)
    return values.length ? Math.max(...values) : null
  })
}
