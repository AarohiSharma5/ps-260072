import { useState } from "react"
import { DomainMap } from "../components/DomainMap"
import { useCities, useSummary } from "../hooks"
import type { NowcastSet, NowcastSetScore } from "../types"

const SETS: NowcastSet[] = ["env", "env+sat", "env+sat+obs"]
const SHORT: Record<NowcastSet, string> = {
  env: "Weather model only",
  "env+sat": "+ INSAT-3DS satellite",
  "env+sat+obs": "+ airport reports (last 3 h)",
}
const COLOR: Record<NowcastSet, string> = { env: "var(--faint)", "env+sat": "var(--cyan)", "env+sat+obs": "var(--amber)" }

const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(d)}%`)
const num = (v: number | null | undefined, d = 2) => (v === null || v === undefined ? "n/a" : v.toFixed(d))

export function ModelPage() {
  const { ready, domains, city, select, error: cityError } = useCities()
  const { summary, error } = useSummary(city)
  const [lead, setLead] = useState("60")

  if (cityError || error) {
    return (
      <main className="page">
        <p className="kicker">Model verification</p>
        <h1>Results are not available.</h1>
        <p>{cityError ?? error}</p>
      </main>
    )
  }
  if (!summary || !city) {
    return (
      <main className="page">
        <p className="kicker">Model verification</p>
        <h1>Reading the results.</h1>
      </main>
    )
  }

  const block = summary.leads[lead]
  const name = summary.meta.city_name
  const base = block.sets["env"]
  const sat = block.sets["env+sat"]
  const full = block.sets["env+sat+obs"]

  return (
    <main className="page cn">
      <p className="kicker">Model verification · {name} · out-of-fold</p>
      <h1>How well does it work?</h1>
      <p className="lede">
        Every score below comes from days the model never saw: {summary.meta.folds}-fold cross-validation in {summary.meta.block_days}-day blocks over{" "}
        {summary.meta.days} days of 2025. The truth is a thunderstorm reported by the airport ({summary.meta.station}).
      </p>

      {ready.length > 1 && (
        <div className="switch-row">
          <div className="switch" role="group" aria-label="City">
            {ready.map((c) => (
              <button key={c.key} type="button" className={c.key === city ? "on" : ""} onClick={() => select(c.key)}>
                {c.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <section>
        <h2>Where it was trained</h2>
        <DomainMap domains={domains} selected={city} onSelect={ready.some((c) => c.key !== city) ? select : undefined} />
      </section>

      <section>
        <dl className="facts inline">
          <dt>Days</dt>
          <dd>{summary.meta.days}</dd>
          <dt>Satellite scenes</dt>
          <dd>{summary.meta.scenes.toLocaleString()}</dd>
          <dt>Season</dt>
          <dd>{summary.meta.season}</dd>
          <dt>Features</dt>
          <dd>
            {summary.meta.feature_sets["env"] ?? "n/a"} weather, {(summary.meta.feature_sets["env+sat"] ?? 0) - (summary.meta.feature_sets["env"] ?? 0)} satellite,{" "}
            {(summary.meta.feature_sets["env+sat+obs"] ?? 0) - (summary.meta.feature_sets["env+sat"] ?? 0)} airport
          </dd>
          <dt>Trained</dt>
          <dd>{summary.meta.trained_at.slice(0, 16).replace("T", " ")} UTC</dd>
        </dl>
      </section>

      <div className="switch-row">
        <div className="switch" role="group" aria-label="Lead time">
          {Object.keys(summary.leads).map((minute) => (
            <button key={minute} type="button" className={minute === lead ? "on" : ""} onClick={() => setLead(minute)}>
              +{minute} min
            </button>
          ))}
        </div>
        <span className="quiet">
          {block.events} storm cases in {block.rows.toLocaleString()} scenes ({pct(block.event_rate, 1)})
        </span>
      </div>

      <section>
        <h2>Scores</h2>
        <table className="metrics">
          <thead>
            <tr>
              <th>Inputs</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
              <th>Skill vs base rate</th>
              <th>Storms caught</th>
              <th>False alerts</th>
              <th>Success index</th>
            </tr>
          </thead>
          <tbody>
            {SETS.map((s) => {
              const r = block.sets[s]
              return (
                <tr key={s}>
                  <th>
                    <span className="cn-dot" style={{ background: COLOR[s] }} />
                    {SHORT[s]}
                  </th>
                  <td>{num(r.roc_auc, 3)}</td>
                  <td>{num(r.pr_auc, 3)}</td>
                  <td>{r.brier_skill === null ? "n/a" : `${r.brier_skill >= 0 ? "+" : ""}${r.brier_skill.toFixed(2)}`}</td>
                  <td>{pct(r.pod)}</td>
                  <td>{pct(r.far)}</td>
                  <td>{num(r.csi, 3)}</td>
                </tr>
              )
            })}
            <tr className="cn-baseline">
              <th>Baseline: storm reported in the last hour</th>
              <td>n/a</td>
              <td>n/a</td>
              <td>n/a</td>
              <td>{pct(block.persistence.pod)}</td>
              <td>{pct(block.persistence.far)}</td>
              <td>{num(block.persistence.csi, 3)}</td>
            </tr>
          </tbody>
        </table>
        <p className="fine">
          Alert thresholds ({num(sat.threshold)} satellite, {num(full.threshold)} with airport reports) were chosen inside the training folds only. Chance level for PR-AUC is{" "}
          {pct(block.event_rate, 1)}.
        </p>
      </section>

      <section>
        <h2>Reading the result honestly</h2>
        <ul className="cn-notes">
          <li>
            The satellite lifts ROC-AUC from {num(base.roc_auc)} to {num(sat.roc_auc)} at +{lead} min, and raises the share of storms caught from {pct(base.pod)} to {pct(sat.pod)}.
          </li>
          <li>
            Even so, {pct(sat.far)} of its alerts are false. A thunderstorm at one airport is rare, and a 35 km satellite pixel cannot tell which cloud will produce it.
          </li>
          <li>
            With the airport's own recent reports the success index is {num(full.csi, 3)}, against {num(block.persistence.csi, 3)} for simply repeating "a storm was reported in the last hour".{" "}
            {full.csi !== null && block.persistence.csi !== null && full.csi > block.persistence.csi
              ? "The model is ahead of that baseline."
              : "The model does not beat that baseline at this lead; its value is the early warning before a storm starts."}
          </li>
        </ul>
      </section>

      <div className="report-grid">
        <section>
          <h2>Hits and misses</h2>
          <p className="fine">At each model's alert threshold, counted over all scenes.</p>
          <div className="cn-conf">
            {(["env+sat", "env+sat+obs"] as NowcastSet[]).map((s) => (
              <Confusion key={s} title={SHORT[s]} score={block.sets[s]} color={COLOR[s]} />
            ))}
          </div>
        </section>
        <section>
          <h2>Are the probabilities honest?</h2>
          <p className="fine">Predicted against observed frequency, on out-of-fold scenes. The diagonal would be perfect.</p>
          <Reliability sets={SETS.map((s) => ({ set: s, bins: block.sets[s].reliability }))} />
        </section>
      </div>
    </main>
  )
}

function Confusion({ title, score, color }: { title: string; score: NowcastSetScore; color: string }) {
  return (
    <table className="matrix">
      <caption style={{ color }}>{title}</caption>
      <thead>
        <tr>
          <th />
          <th>Alert</th>
          <th>No alert</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Storm followed</th>
          <td>{score.tp}</td>
          <td>{score.fn}</td>
        </tr>
        <tr>
          <th>No storm</th>
          <td>{score.fp}</td>
          <td>{score.tn}</td>
        </tr>
      </tbody>
    </table>
  )
}

function Reliability({ sets }: { sets: { set: NowcastSet; bins: { mean_pred: number; obs_freq: number; n: number }[] }[] }) {
  const size = 280
  const pad = 36
  const maxV = Math.max(0.1, ...sets.flatMap((s) => s.bins.flatMap((b) => [b.mean_pred, b.obs_freq]))) * 1.1
  const x = (v: number) => pad + (v / maxV) * (size - pad - 10)
  const y = (v: number) => size - pad - (v / maxV) * (size - pad - 10)
  const ticks = [0, maxV / 2, maxV]
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="cn-rel" role="img" aria-label="Reliability diagram">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad} x2={size - 10} y1={y(t)} y2={y(t)} className="cn-grid" />
          <text x={pad - 6} y={y(t) + 4} textAnchor="end" className="cn-axis">
            {Math.round(t * 100)}%
          </text>
          <text x={x(t)} y={size - pad + 16} textAnchor="middle" className="cn-axis">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(maxV)} y2={y(maxV)} className="cn-thresh" />
      {sets.map(({ set, bins }) => (
        <g key={set}>
          <polyline points={bins.map((b) => `${x(b.mean_pred)},${y(b.obs_freq)}`).join(" ")} fill="none" stroke={COLOR[set]} strokeWidth={1.8} />
          {bins.map((b) => (
            <circle key={b.mean_pred} cx={x(b.mean_pred)} cy={y(b.obs_freq)} r={3} fill={COLOR[set]} />
          ))}
        </g>
      ))}
      <text x={size / 2} y={size - 4} textAnchor="middle" className="cn-axis">
        predicted probability
      </text>
      <text x={10} y={size / 2} className="cn-axis" transform={`rotate(-90 10 ${size / 2})`} textAnchor="middle">
        observed frequency
      </text>
    </svg>
  )
}
