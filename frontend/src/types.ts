export type DataMode = "DEMO" | "HISTORICAL" | "SEVIR" | "LIVE" | "UNAVAILABLE"
export type Hazard = "thunderstorm" | "lightning"

export interface Dataset {
  id: string
  label: string
  status: DataMode
  note: string
  timestamp_label?: string | null
  freshness?: string | null
}

export interface Domain {
  name: string
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
  n_lat: number
  n_lon: number
  center_lat: number
  center_lon: number
  cell_km_lat: number
  cell_km_lon: number
}

export interface CaseScore {
  n: number
  mean_predicted: number
  observed_fraction: number
  brier: number | null
  f1: number
  roc_auc: number | null
}

export interface CaseSummary {
  id: string
  title: string
  summary: string
  data_mode: DataMode
  in_training_set: boolean
  analysis_index: number
  clock: string
  clock_kind: "utc" | "simulated"
  step_min: number
  scores?: { note: string; targets: Record<string, CaseScore> }
}

export interface GridCell {
  grid_id: number
  row: number
  col: number
  latitude: number
  longitude: number
  polygon: number[][]
}

export type Series = (number | null)[]

export interface Analysis {
  offset_min: number
  clock: string
  reflectivity: Series | null
  vil: Series | null
  ir_bt: Series | null
  temperature: Series | null
  rh: Series | null
  pressure: Series | null
  wind_speed: Series | null
  wind_direction: Series | null
  wind_u: Series | null
  wind_v: Series | null
  cape: Series | null
  cin: Series | null
  shear: Series | null
  k_index: Series | null
  total_totals: Series | null
  convective_precip: Series | null
  lightning_count: Series | null
  lightning_density: Series | null
  refl_trend: Series | null
  ctt_cooling: Series | null
  cp_trend: Series | null
  lightning_rate_change: Series | null
  motion_u: Series | null
  motion_v: Series | null
  convergence: Series | null
  thunderstorm_indicator: Series | null
  lightning_indicator: Series | null
}

export type HazardLeads = Partial<Record<Hazard, Record<string, Series>>>

export interface Scenario {
  data_mode: DataMode
  disclaimer: string
  source_name: string
  model_version: string
  model_trained_at: string
  product_generated_at: string
  case: CaseSummary
  domain: Domain
  step_min: number
  step_text: string
  hazards: Hazard[]
  leads_min: number[]
  panel_leads_min: number[]
  timeline_min: number[]
  thresholds: { watch: number; high: number; severe: number; status: string; rule: string }
  confidence_note: string
  label_definitions: Record<string, string>
  grid: GridCell[]
  places: { name: string; lat: number; lon: number }[]
  analysis: Analysis
  history: {
    offsets_min: number[]
    clocks: string[]
    reflectivity: Series[] | null
    vil: Series[] | null
    ir_bt: Series[] | null
    lightning_count: Series[] | null
    convective_precip: Series[] | null
    cape: Series[] | null
  }
  prediction: HazardLeads
  extrapolation: HazardLeads | null
  persistence: Partial<Record<Hazard, Series>>
  verification: {
    note: string
    offsets_min: number[]
    clocks: string[]
    reflectivity: Series[] | null
    vil: Series[] | null
    lightning_count: Series[] | null
    convective_precip: Series[] | null
    thunderstorm: Series[] | null
    lightning_event: Series[] | null
  }
  case_scores: { note: string; targets: Record<string, CaseScore> }
  datasets: Dataset[]
  feature_names: string[]
  feature_meta: Record<string, { label: string; unit: string; group: string }>
  layers: { id: string; label: string; unit: string; available: boolean; status: DataMode }[]
}

export interface Status {
  system: string
  problem_statement: string
  model_version: string
  model_trained_at: string
  data_mode: DataMode
  interactive_mode: DataMode
  disclaimer: string
  source_name: string
  domain: { name: string; lat_min: number; lat_max: number; lon_min: number; lon_max: number; n_lat: number; n_lon: number }
  step_min: number
  leads_min: number[]
  hazards: Hazard[]
  skipped_targets: string[]
  live_feeds_connected: boolean
  datasets: Dataset[]
  model_ready: boolean
  period: { first: string; last: string } | null
}

export interface OperatingMetrics {
  threshold: number
  precision: number
  recall: number
  f1: number
  roc_auc: number | null
  average_precision: number | null
  brier: number | null
  confusion: { tn: number; fp: number; fn: number; tp: number }
  roc_auc_defined: boolean
}

export interface Importance {
  feature: string
  label: string
  unit: string
  group: string
  gain: number
  gain_share: number
}

export interface TargetMetrics {
  hazard: string
  lead_min: number
  n_trees: number
  support: { n: number; positives: number; negatives: number; positive_rate: number | null }
  model: OperatingMetrics
  persistence: OperatingMetrics
  extrapolation: OperatingMetrics | null
  reliability: { bin_lo: number; bin_hi: number; count: number; mean_predicted: number | null; fraction_positive: number | null }[]
  feature_importance: Importance[]
}

export interface MetricsReport {
  trained_at: string
  model_version: string
  data_mode: DataMode
  step_min: number
  leads_min: number[]
  hazards: Hazard[]
  disclaimer: string
  label_definitions: Record<string, string>
  threshold_note: string
  independence_note: string
  extrapolation_available: boolean
  targets: Record<string, TargetMetrics>
}

export interface Explanation {
  data_mode: DataMode
  grid_id: number
  hazard: string
  lead_min: number
  probability: number
  bias_log_odds: number
  contributions_are: string
  disclaimer: string
  features: { feature: string; label: string; unit: string; value: number | null; contribution: number; statement: string }[]
}

export interface Methodology {
  title: string
  data_mode: DataMode
  disclaimer: string
  source_name: string | null
  steps: { id: string; title: string; body: string }[]
  split: {
    scheme: string
    train_cases: number
    val_cases: number
    test_cases: number
    train_rows: number
    val_rows: number
    test_rows: number
    first_train_case: string
    last_test_case: string
    leakage_controls: string[]
  }
  features: { id: string; label: string; unit: string; group: string }[]
  baselines: { persistence: string; extrapolation: string }
  thresholds: { watch: number; high: number; severe: number; status: string }
  hazards: Hazard[]
  skipped_targets: string[]
  leads_min: number[]
  step_min: number
  scaling: string[]
  limitations: string[]
}

export type BaseLayer =
  | "thunderstorm"
  | "lightning"
  | "extrapolation"
  | "reflectivity"
  | "vil"
  | "ir_bt"
  | "lightning_density"
  | "convective_precip"
  | "temperature"
  | "rh"
  | "cape"
  | "cin"
  | "shear"
  | "k_index"
  | "total_totals"

export interface Overlays {
  grid: boolean
  lightning: boolean
  wind: boolean
  motion: boolean
  verify: boolean
  places: boolean
}
