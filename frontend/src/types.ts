export type NowcastSet = "env" | "env+sat" | "env+sat+obs"

export interface NowcastSetScore {
  label: string
  roc_auc: number | null
  pr_auc: number | null
  brier_skill: number | null
  pod: number | null
  far: number | null
  csi: number | null
  threshold: number | null
  tp: number
  fp: number
  fn: number
  tn: number
  reliability: { mean_pred: number; obs_freq: number; n: number }[]
}

export interface NowcastSummary {
  meta: {
    city: string
    city_name: string
    station: string
    season: string
    scenes: number
    days: number
    first: string
    last: string
    folds: number
    block_days: number
    label: string
    satellite: string
    environment: string
    trained_at: string
    feature_sets: Record<string, number>
    lat: number
    lon: number
  }
  set_labels: Record<NowcastSet, string>
  leads: Record<
    string,
    {
      events: number
      rows: number
      event_rate: number
      sets: Record<NowcastSet, NowcastSetScore>
      persistence: { pod: number | null; far: number | null; csi: number | null }
    }
  >
  importance: Record<string, { feature: string; label: string; group: "satellite" | "environment"; gain: number }[]>
  notes: string[]
}

export interface NowcastDayInfo {
  day: string
  scenes: number
  storm_scenes: number
  reports: number
  coldest_k: number | null
}

export interface NowcastDay {
  day: string
  times: string[]
  series: { ctt_min: (number | null)[]; cold220: (number | null)[]; ctp_min: (number | null)[]; cloudiness: (number | null)[]; cape: (number | null)[] }
  probability: Record<NowcastSet, Record<string, (number | null)[]>>
  truth: Record<string, number[]>
  reports: { time: string; metar: string }[]
}

export interface NowcastCity {
  key: string
  name: string
  station: string
  lat: number
  lon: number
  season: string
  ready: boolean
}

export interface LiveLead {
  probability: number
  threshold: number
  base_rate: number
  alert: boolean
}

export interface LiveForecast {
  city: string
  city_name: string
  station: string
  generated_at: string
  scene_time: string
  scene_age_min: number
  scenes_used: string[]
  models: Record<"env_sat" | "env_sat_obs", { label: string; leads: Record<string, LiveLead> }>
  drivers: Record<string, number | null>
  airport: {
    storm_reported_last_hour: boolean
    storm_reported_last_3h: boolean
    latest_report: { time: string; metar: string } | null
    storm_reports_since_scene: { time: string; metar: string }[]
  }
  quality: { missing_feature_share: number }
  notes: string[]
  stale?: boolean
  stale_reason?: string
}

export interface NowcastDomain {
  city: string
  name: string
  station: string
  lat: number
  lon: number
  window: [number, number, number, number]
  pixels: [number, number][]
  pixel_spacing_km: number
  ready: boolean
  season: string
}
