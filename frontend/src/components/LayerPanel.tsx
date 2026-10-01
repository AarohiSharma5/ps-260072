import type { BaseLayer, Hazard, Overlays, Scenario } from "../types"
import { formatNumber, maxOf, meanOf, riskClass, riskLevel } from "../format"
import { activeModelLead, combinedProbability } from "../mapdata"

interface LayerPanelProps {
  scenario: Scenario
  base: BaseLayer
  hazard: Hazard
  offset: number
  overlays: Overlays
  opacity: number
  onBase: (base: BaseLayer) => void
  onHazard: (hazard: Hazard) => void
  onOverlays: (overlays: Overlays) => void
  onOpacity: (opacity: number) => void
  onPreset: (preset: "analysis" | "nowcast") => void
}

const OBSERVED: { id: BaseLayer; label: string; layer: string }[] = [
  { id: "reflectivity", label: "Radar reflectivity", layer: "reflectivity" },
  { id: "vil", label: "Radar VIL", layer: "vil" },
  { id: "ir_bt", label: "Brightness temperature", layer: "ir_bt" },
  { id: "lightning_density", label: "Lightning density", layer: "lightning_density" },
  { id: "convective_precip", label: "Convective precipitation", layer: "convective_precip" },
  { id: "temperature", label: "Temperature", layer: "temperature" },
  { id: "rh", label: "Humidity", layer: "rh" },
  { id: "cape", label: "CAPE", layer: "cape" },
  { id: "cin", label: "CIN", layer: "cin" },
  { id: "shear", label: "Bulk shear", layer: "shear" },
  { id: "k_index", label: "K index", layer: "k_index" },
  { id: "total_totals", label: "Totals-Totals", layer: "total_totals" },
]

export function LayerPanel(props: LayerPanelProps) {
  const { scenario } = props
  const lead = activeModelLead(props.offset, scenario)
  const combined = combinedProbability(scenario, lead)
  const counts = countRisk(combined, scenario.thresholds)
  const layerStatus = new Map(scenario.layers.map((layer) => [layer.id, layer]))
  const hasLightning = scenario.hazards.includes("lightning")
  const hasExtrap = scenario.extrapolation !== null
  const hasRadar = layerStatus.get("reflectivity")?.available ?? false
  const forecast: { id: BaseLayer; label: string; available: boolean; note: string }[] = [
    { id: "thunderstorm", label: "Thunderstorm probability", available: scenario.hazards.includes("thunderstorm"), note: scenario.data_mode === "DEMO" ? "" : "proxy label" },
    { id: "lightning", label: "Lightning probability", available: hasLightning, note: hasLightning ? "" : "UNAVAILABLE" },
    { id: "extrapolation", label: "Motion extrapolation", available: hasExtrap, note: hasExtrap ? props.hazard : "needs radar" },
  ]

  return (
    <aside className="float-panel left-panel">
      <div className="panel-head">
        <p className="kicker">
          {scenario.domain.name} · {scenario.data_mode}
        </p>
        <h1>{scenario.case.title}</h1>
        <p className="quiet">{scenario.case.summary}</p>
      </div>
      <div className="preset-row">
        <button type="button" onClick={() => props.onPreset("analysis")}>
          Analysis
        </button>
        <button type="button" className="primary" onClick={() => props.onPreset("nowcast")}>
          Nowcast
        </button>
      </div>
      <div className="risk-tally" aria-label="Prototype risk counts">
        <div className="tally-label">Prototype risk at +{lead} min</div>
        {(
          [
            ["SEVERE RISK", counts.severe],
            ["HIGH RISK", counts.high],
            ["WATCH", counts.watch],
            ["NORMAL", counts.normal],
          ] as const
        ).map(([level, count]) => (
          <div key={level} className="tally-row">
            <span className={`risk-pill ${riskClass(level)}`}>{level}</span>
            <span className="tally-count">{count}</span>
          </div>
        ))}
        <p className="fine">
          Highest available hazard probability. Cuts at {scenario.thresholds.watch}, {scenario.thresholds.high}, {scenario.thresholds.severe}.
          Not an official warning standard.
          {props.offset === 0 ? ` Timeline is on the analysis, so this tally uses +${lead} min.` : ""}
        </p>
      </div>
      <section>
        <h2>Forecast field</h2>
        {forecast.map((item) => (
          <button
            key={item.id}
            type="button"
            disabled={!item.available}
            className={props.base === item.id ? "layer-option on" : "layer-option"}
            onClick={() => {
              props.onBase(item.id)
              if (item.id === "thunderstorm" || item.id === "lightning") props.onHazard(item.id)
            }}
          >
            <span>{item.label}</span>
            {item.note && <span className="option-note">{item.note}</span>}
          </button>
        ))}
      </section>
      <section>
        <h2>Observed field</h2>
        {OBSERVED.map((item) => {
          const layer = layerStatus.get(item.layer)
          const available = layer?.available ?? false
          return (
            <button
              key={item.id}
              type="button"
              disabled={!available}
              className={props.base === item.id ? "layer-option on" : "layer-option"}
              onClick={() => props.onBase(item.id)}
            >
              <span>{item.label}</span>
              <span className="option-note">{layer?.status ?? "UNAVAILABLE"}</span>
            </button>
          )
        })}
      </section>
      <section>
        <h2>Overlays</h2>
        <Toggle label="Grid" on={props.overlays.grid} set={(grid) => props.onOverlays({ ...props.overlays, grid })} />
        <Toggle label="Lightning" on={props.overlays.lightning} disabled={!hasLightning} set={(lightning) => props.onOverlays({ ...props.overlays, lightning })} />
        <Toggle label="Wind" on={props.overlays.wind} disabled={!(layerStatus.get("wind")?.available ?? false)} set={(wind) => props.onOverlays({ ...props.overlays, wind })} />
        <Toggle label="Storm movement" on={props.overlays.motion} disabled={!hasRadar} set={(motion) => props.onOverlays({ ...props.overlays, motion })} />
        <Toggle label="Verifying later frames" on={props.overlays.verify} set={(verify) => props.onOverlays({ ...props.overlays, verify })} />
        <Toggle label="Place names" on={props.overlays.places} set={(places) => props.onOverlays({ ...props.overlays, places })} />
      </section>
      <section>
        <h2>Layer opacity</h2>
        <input className="opacity" type="range" min={0.2} max={1} step={0.02} value={props.opacity} onChange={(event) => props.onOpacity(Number(event.target.value))} aria-label="Layer opacity" />
      </section>
      <section className="sources">
        <h2>Data status</h2>
        <p className="fine">
          Analysis {scenario.analysis.clock} · step {scenario.step_text} · model {scenario.model_version}. No live feed is connected.
        </p>
        <ul>
          {scenario.datasets.map((dataset) => (
            <li key={dataset.id}>
              <span className={`status-dot status-${dataset.status.toLowerCase()}`}>{dataset.status}</span>
              <span>
                <strong>{dataset.label}</strong>
                <em>{dataset.note}</em>
                {dataset.timestamp_label && <em>Latest frame {dataset.timestamp_label}</em>}
              </span>
            </li>
          ))}
        </ul>
      </section>
      <p className="fine readout">
        {scenario.analysis.reflectivity
          ? `Domain mean reflectivity ${formatNumber(meanOf(scenario.analysis.reflectivity), 1)} dBZ · peak ${formatNumber(maxOf(scenario.analysis.reflectivity), 0)} dBZ`
          : scenario.analysis.cape
            ? `Domain mean CAPE ${formatNumber(meanOf(scenario.analysis.cape), 0)} J/kg · peak ${formatNumber(maxOf(scenario.analysis.cape), 0)} J/kg`
            : "No summary field available."}
      </p>
    </aside>
  )
}

function Toggle({ label, on, set, disabled }: { label: string; on: boolean; set: (on: boolean) => void; disabled?: boolean }) {
  return (
    <label className={disabled ? "toggle disabled" : "toggle"}>
      <input type="checkbox" checked={on && !disabled} disabled={disabled} onChange={(event) => set(event.target.checked)} />
      <span>{label}</span>
      {disabled && <span className="option-note">UNAVAILABLE</span>}
    </label>
  )
}

function countRisk(values: (number | null)[] | null, thresholds: Scenario["thresholds"]) {
  const counts = { severe: 0, high: 0, watch: 0, normal: 0 }
  if (!values) return counts
  values.forEach((value) => {
    const level = riskLevel(value, thresholds)
    if (level === "SEVERE RISK") counts.severe += 1
    else if (level === "HIGH RISK") counts.high += 1
    else if (level === "WATCH") counts.watch += 1
    else if (level === "NORMAL") counts.normal += 1
  })
  return counts
}
