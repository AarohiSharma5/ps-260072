import { useEffect, useState } from "react"
import { api } from "../api"
import { activeModelLead, hazardForBase } from "../mapdata"
import type { BaseLayer, Explanation, Hazard, Scenario, Series } from "../types"
import { confidenceMargin, coordinates, formatNumber, formatSigned, riskClass, riskLevel } from "../format"

interface DetailProps {
  scenario: Scenario
  gridId: number
  base: BaseLayer
  hazard: Hazard
  offset: number
  onClose: () => void
}

export function DetailPanel({ scenario, gridId, base, hazard, offset, onClose }: DetailProps) {
  const cell = scenario.grid[gridId]
  const a = scenario.analysis
  const lead = activeModelLead(offset, scenario)
  const explainHazard = hazardForBase(base, hazard, scenario)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [explainError, setExplainError] = useState<string | null>(null)

  useEffect(() => {
    let cancel = false
    setExplanation(null)
    setExplainError(null)
    api
      .explanation(scenario.case.id, gridId, explainHazard, lead)
      .then((result) => !cancel && setExplanation(result))
      .catch((reason: unknown) => !cancel && setExplainError(reason instanceof Error ? reason.message : "Explanation unavailable"))
    return () => {
      cancel = true
    }
  }, [scenario.case.id, gridId, explainHazard, lead])

  const panelProbs = scenario.hazards.flatMap((h) => scenario.panel_leads_min.map((m) => scenario.prediction[h]?.[String(m)]?.[gridId] ?? null))
  const margin = confidenceMargin(panelProbs.filter((v): v is number => v !== null))
  const activeProbability = scenario.prediction[explainHazard]?.[String(lead)]?.[gridId] ?? null
  const level = riskLevel(activeProbability, scenario.thresholds)
  const at = (name: keyof typeof a): number | null => {
    const series = a[name]
    return Array.isArray(series) ? series[gridId] ?? null : null
  }
  const step = scenario.step_text
  const hasLightning = scenario.hazards.includes("lightning")

  return (
    <aside className="float-panel detail-panel">
      <div className="detail-head">
        <div>
          <p className="kicker">
            Grid {cell.grid_id} · {scenario.data_mode}
          </p>
          <h2>{coordinates(cell.latitude, cell.longitude)}</h2>
        </div>
        <button type="button" className="text-button" onClick={onClose}>
          Close
        </button>
      </div>
      <div className={`risk-pill large ${riskClass(level)}`}>{level}</div>
      <p className="fine">
        Risk uses the +{lead} min {explainHazard} probability{offset === 0 ? " while the timeline is on the analysis" : ""}. Prototype threshold.
      </p>

      <section>
        <h3>Thunderstorm</h3>
        {scenario.data_mode === "HISTORICAL" && <p className="fine">{scenario.label_definitions.thunderstorm}</p>}
        <ProbList scenario={scenario} gridId={gridId} hazard="thunderstorm" active={lead} highlight={explainHazard === "thunderstorm"} />
      </section>
      <section>
        <h3>Lightning</h3>
        {hasLightning ? (
          <ProbList scenario={scenario} gridId={gridId} hazard="lightning" active={lead} highlight={explainHazard === "lightning"} />
        ) : (
          <p className="fine">UNAVAILABLE. This dataset has no lightning observations, so no lightning model was trained.</p>
        )}
      </section>
      <p className="fine">
        Prototype confidence {formatNumber(margin, 2)}. {scenario.confidence_note}
      </p>

      <section>
        <h3>Atmospheric conditions</h3>
        <p className="fine">
          Analysis {a.clock} · {scenario.data_mode}
        </p>
        <dl className="facts">
          <Fact label="Temperature" value={formatNumber(at("temperature"), 1)} unit="°C" />
          <Fact label="Humidity" value={formatNumber(at("rh"), 0)} unit="%" />
          <Fact label="Pressure" value={formatNumber(at("pressure"), 1)} unit="hPa" />
          <Fact label="Wind speed" value={formatNumber(at("wind_speed"), 1)} unit="m/s" />
          <Fact label="Wind direction" value={direction(at("wind_direction"))} unit="" />
          <Fact label="CAPE" value={formatNumber(at("cape"), 0)} unit="J/kg" />
          <Fact label="CIN" value={formatNumber(at("cin"), 0)} unit="J/kg" />
          <Fact label="Wind shear" value={formatNumber(at("shear"), 1)} unit="m/s" />
          {a.k_index && <Fact label="K index" value={formatNumber(at("k_index"), 1)} unit="°C" />}
          {a.total_totals && <Fact label="Totals-Totals" value={formatNumber(at("total_totals"), 1)} unit="°C" />}
          {a.convective_precip && <Fact label="Convective precipitation" value={formatNumber(at("convective_precip"), 2)} unit="mm/h" />}
        </dl>
      </section>

      <section>
        <h3>Recent evolution</h3>
        <p className="fine">
          Previous {scenario.history.offsets_min.length - 1} steps of {step}, ending at analysis. {scenario.data_mode}
        </p>
        <Spark label="Reflectivity" unit="dBZ" values={series(scenario.history.reflectivity, gridId)} />
        <Spark label="Brightness temperature" unit="K" values={series(scenario.history.ir_bt, gridId)} />
        <Spark label="Lightning count" unit="flashes" values={series(scenario.history.lightning_count, gridId)} />
        {scenario.history.convective_precip && <Spark label="Convective precipitation" unit="mm/h" values={series(scenario.history.convective_precip, gridId)} />}
        {scenario.history.cape && <Spark label="CAPE" unit="J/kg" values={series(scenario.history.cape, gridId)} />}
        <dl className="facts">
          <Fact label="Reflectivity trend" value={formatSigned(at("refl_trend"), 1)} unit={`dBZ / ${step}`} />
          <Fact label="Cloud-top cooling" value={formatSigned(at("ctt_cooling"), 1)} unit={`K / ${step}`} />
          <Fact label="Lightning trend" value={formatSigned(at("lightning_rate_change"), 1)} unit={`flashes / ${step}`} />
          {a.cp_trend && <Fact label="Convective precipitation trend" value={formatSigned(at("cp_trend"), 2)} unit={`mm/h / ${step}`} />}
          <Fact label="Storm movement" value={movement(at("motion_u"), at("motion_v"))} unit="" />
        </dl>
      </section>

      <section>
        <h3>Why this prediction?</h3>
        <p className="fine">
          {explainHazard} model at +{lead} min. {explanation?.contributions_are}
        </p>
        {explainError && <p className="fine">{explainError}</p>}
        {!explanation && !explainError && <p className="fine">Reading model contributions.</p>}
        {explanation && (
          <ul className="contribs">
            {explanation.features.map((feature) => (
              <li key={feature.feature}>
                <div className="contrib-top">
                  <span>{feature.label}</span>
                  <span className="mono">
                    {formatNumber(feature.value, 2)} {feature.value === null ? "" : feature.unit}
                  </span>
                </div>
                <div className="contrib-bar">
                  <span style={{ width: `${Math.min(100, Math.abs(feature.contribution) * 28)}%` }} data-sign={feature.contribution >= 0 ? "pos" : "neg"} />
                </div>
                <div className="contrib-note">
                  {feature.statement}
                  <em>
                    {feature.contribution > 0 ? "+" : ""}
                    {feature.contribution.toFixed(2)} log-odds
                  </em>
                </div>
              </li>
            ))}
          </ul>
        )}
        <p className="fine">{explanation?.disclaimer ?? "Contributions attribute the model score. They do not prove a physical cause."}</p>
      </section>
    </aside>
  )
}

function ProbList({ scenario, gridId, hazard, active, highlight }: { scenario: Scenario; gridId: number; hazard: Hazard; active: number; highlight: boolean }) {
  const extrap = scenario.extrapolation?.[hazard]?.[String(active)]?.[gridId]
  return (
    <div className="prob-list">
      {scenario.leads_min.map((minute) => {
        const value = scenario.prediction[hazard]?.[String(minute)]?.[gridId]
        const width = value === undefined || value === null ? 0 : Math.max(0, Math.min(1, value)) * 100
        return (
          <div key={minute} className={minute === active && highlight ? "prob-row active" : "prob-row"}>
            <span>+{minute}</span>
            <span className="prob-track">
              <span style={{ width: `${width}%` }} />
            </span>
            <span className="mono">{formatNumber(value, 2)}</span>
          </div>
        )
      })}
      <p className="fine">
        Persistence {formatNumber(scenario.persistence[hazard]?.[gridId], 0)} · extrapolation +{active} {extrap === undefined ? "Unavailable" : formatNumber(extrap, 2)}
      </p>
    </div>
  )
}

function Fact({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>
        {value}
        {value !== "Unavailable" && unit ? <span> {unit}</span> : null}
      </dd>
    </>
  )
}

function Spark({ label, unit, values }: { label: string; unit: string; values: number[] | null }) {
  if (!values || values.length < 2) {
    return (
      <div className="spark-row">
        <span>{label}</span>
        <span>Unavailable</span>
      </div>
    )
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const path = values
    .map((value, index) => `${index === 0 ? "M" : "L"}${((index / (values.length - 1)) * 100).toFixed(1)} ${(26 - ((value - min) / span) * 20).toFixed(1)}`)
    .join(" ")
  return (
    <div className="spark-row">
      <span>{label}</span>
      <svg viewBox="0 0 100 32" className="spark" aria-hidden="true">
        <path d={path} />
      </svg>
      <span className="mono">
        {formatNumber(values[values.length - 1], 1)} {unit}
      </span>
    </div>
  )
}

function series(frames: Series[] | null, gridId: number): number[] | null {
  if (!frames) return null
  const values = frames.map((frame) => frame[gridId])
  if (values.some((v) => v === null || v === undefined)) return null
  return values as number[]
}

function direction(value: number | null): string {
  return value === null ? "Unavailable" : `from ${value.toFixed(0)}°`
}

function movement(u: number | null, v: number | null): string {
  if (u === null || v === null) return "Unavailable"
  const speed = Math.hypot(u, v) * 3.6
  const heading = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360
  return `toward ${heading.toFixed(0)}° at ${speed.toFixed(0)} km/h`
}
