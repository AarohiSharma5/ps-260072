import { useEffect, useMemo, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { api } from "../api"
import type { LiveForecast, NowcastCity, NowcastDay, NowcastDayInfo, NowcastSet, NowcastSummary } from "../types"

const SETS: NowcastSet[] = ["env", "env+sat", "env+sat+obs"]
const SHORT: Record<NowcastSet, string> = {
  env: "Weather model only",
  "env+sat": "+ INSAT-3DS satellite",
  "env+sat+obs": "+ airport reports (last 3 h)",
}
const COLOR: Record<NowcastSet, string> = {
  env: "var(--faint)",
  "env+sat": "var(--cyan)",
  "env+sat+obs": "var(--amber)",
}

const pct = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "n/a" : `${(value * 100).toFixed(digits)}%`
const num = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined ? "n/a" : value.toFixed(digits)

function istLabel(iso: string): string {
  const ms = Date.parse(iso) + 5.5 * 3600 * 1000
  const d = new Date(ms)
  return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`
}

function istDate(iso: string): string {
  const d = new Date(Date.parse(iso) + 5.5 * 3600 * 1000)
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
}

export function NowcastPage() {
  const [params, setParams] = useSearchParams()
  const [cities, setCities] = useState<NowcastCity[]>([])
  const [liveOn, setLiveOn] = useState(false)
  const [defaultCity, setDefaultCity] = useState("chennai")
  const [summary, setSummary] = useState<NowcastSummary | null>(null)
  const [days, setDays] = useState<NowcastDayInfo[]>([])
  const [day, setDay] = useState<string | null>(null)
  const [detail, setDetail] = useState<NowcastDay | null>(null)
  const [lead, setLead] = useState("60")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .nowcastCities()
      .then((r) => {
        setCities(r.cities)
        setLiveOn(r.live)
        setDefaultCity(r.default)
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "The API did not respond."))
  }, [])

  const requested = params.get("city")
  const city = cities.find((c) => c.key === requested && c.ready)?.key ?? (cities.find((c) => c.key === defaultCity && c.ready)?.key ?? cities.find((c) => c.ready)?.key ?? null)

  useEffect(() => {
    if (!city) return
    let cancel = false
    setSummary(null)
    setDays([])
    setDay(null)
    setDetail(null)
    setError(null)
    Promise.all([api.nowcastSummary(city), api.nowcastDays(city)])
      .then(([s, d]) => {
        if (cancel) return
        setSummary(s)
        setDays(d.days)
        setDay(d.days[0]?.day ?? null)
      })
      .catch((reason: unknown) => !cancel && setError(reason instanceof Error ? reason.message : "Results did not load."))
    return () => {
      cancel = true
    }
  }, [city])

  useEffect(() => {
    if (!day || !city) return
    let cancel = false
    setDetail(null)
    api
      .nowcastDay(city, day)
      .then((next) => !cancel && setDetail(next))
      .catch((reason: unknown) => !cancel && setError(reason instanceof Error ? reason.message : "Day did not load."))
    return () => {
      cancel = true
    }
  }, [city, day])

  if (error) {
    return (
      <main className="page">
        <p className="kicker">Thunderstorm nowcast</p>
        <h1>Results are not available.</h1>
        <p>{error}</p>
        <pre className="cn-code">{`cd backend
python -m app.nowcast_train --city chennai
python -m app.nowcast_report --city chennai`}</pre>
      </main>
    )
  }
  if (!summary || !city) {
    return (
      <main className="page">
        <p className="kicker">Thunderstorm nowcast</p>
        <h1>Loading results.</h1>
      </main>
    )
  }
  const cityName = summary.meta.city_name

  const block = summary.leads[lead]
  const base = block.sets["env"]
  const sat = block.sets["env+sat"]
  const full = block.sets["env+sat+obs"]
  const gain = (sat.roc_auc ?? 0) - (base.roc_auc ?? 0)

  return (
    <main className="page cn">
      <p className="kicker">{cityName} · INSAT-3DS satellite · cross-validated on 2025</p>
      <h1>Thunderstorm nowcast for {cityName}.</h1>
      <p className="lede">
        A probability that the airport reports thunder within the next 30, 60 or 90 minutes. Inputs are cloud tops from ISRO's INSAT-3DS (via MOSDAC) and a
        weather-model environment. The truth is what the airport actually reported.
      </p>

      {cities.filter((c) => c.ready).length > 1 && (
        <div className="switch-row">
          <div className="switch" role="group" aria-label="City">
            {cities
              .filter((c) => c.ready)
              .map((c) => (
                <button key={c.key} type="button" className={c.key === city ? "on" : ""} onClick={() => setParams({ city: c.key })}>
                  {c.name}
                </button>
              ))}
          </div>
          <span className="quiet">
            Airport {summary.meta.station} · {summary.meta.season}
          </span>
        </div>
      )}

      {liveOn && <LiveCard city={city} cityName={cityName} />}

      <div className="switch-row">
        <div className="switch" role="group" aria-label="Lead time">
          {Object.keys(summary.leads).map((minute) => (
            <button key={minute} type="button" className={minute === lead ? "on" : ""} onClick={() => setLead(minute)}>
              +{minute} min
            </button>
          ))}
        </div>
        <span className="quiet">
          {summary.meta.days} days · {summary.meta.scenes.toLocaleString()} satellite scenes · {block.events} storm cases at +{lead} min
        </span>
      </div>

      <section className="cn-cards">
        <div className="cn-card">
          <span className="cn-card-label">Weather model alone</span>
          <span className="cn-card-value">{num(base.roc_auc)}</span>
          <span className="cn-card-note">ROC-AUC</span>
        </div>
        <div className="cn-card cn-card-hot">
          <span className="cn-card-label">With INSAT-3DS satellite</span>
          <span className="cn-card-value">{num(sat.roc_auc)}</span>
          <span className="cn-card-note">
            ROC-AUC · {gain >= 0 ? "+" : ""}
            {gain.toFixed(2)} from the satellite
          </span>
        </div>
        <div className="cn-card">
          <span className="cn-card-label">Plus airport reports</span>
          <span className="cn-card-value">{num(full.roc_auc)}</span>
          <span className="cn-card-note">ROC-AUC · storms already in progress</span>
        </div>
        <div className="cn-card">
          <span className="cn-card-label">Satellite model catches</span>
          <span className="cn-card-value">{pct(sat.pod)}</span>
          <span className="cn-card-note">of storms · {pct(sat.far)} of alerts are false</span>
        </div>
      </section>

      <section>
        <h2>Does the satellite help?</h2>
        <p className="fine">
          Out-of-fold scores: every {summary.meta.block_days}-day block was predicted by a model that never saw it. Alert thresholds were chosen inside the training
          folds only.
        </p>
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
            {SETS.map((name) => {
              const r = block.sets[name]
              return (
                <tr key={name}>
                  <th>
                    <span className="cn-dot" style={{ background: COLOR[name] }} />
                    {SHORT[name]}
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
          Chance level for PR-AUC is the storm rate, {pct(block.event_rate, 1)}. The satellite adds the most when a storm has not started yet, which is the hard case.
          Once a storm is in progress at the airport, "it is still there" is hard to beat.
        </p>
      </section>

      <section>
        <h2>Replay a day</h2>
        <p className="fine">
          Probabilities are out-of-fold: this day was never in the training data of the model that scored it. Shaded bands mark times when a storm was reported within the next
          +{lead} minutes.
        </p>
        <label className="cn-pick">
          <span>Day</span>
          <select value={day ?? ""} onChange={(event) => setDay(event.target.value)}>
            {days.map((d) => (
              <option key={d.day} value={d.day}>
                {d.day} · {d.reports > 0 ? `${d.reports} storm reports` : "no storm reported"}
              </option>
            ))}
          </select>
        </label>
        {detail ? <DayChart detail={detail} lead={lead} /> : <p className="quiet">Loading the day.</p>}
      </section>

      <section>
        <h2>What the +{lead} min model relies on</h2>
        <p className="fine">Share of XGBoost split gain. The satellite features lead because cold, high cloud tops are the physical signature of deep convection.</p>
        <ol className="importance">
          {(summary.importance[lead] ?? []).slice(0, 12).map((row) => {
            const max = summary.importance[lead][0]?.gain || 1
            return (
              <li key={row.feature}>
                <span>{row.label}</span>
                <span className="bar">
                  <span style={{ width: `${(row.gain / max) * 100}%`, background: row.group === "satellite" ? "var(--cyan)" : "var(--faint)" }} />
                </span>
                <span className="mono">{(row.gain * 100).toFixed(1)}%</span>
              </li>
            )
          })}
        </ol>
        <p className="fine">
          <span className="cn-dot" style={{ background: "var(--cyan)" }} />
          satellite &nbsp;
          <span className="cn-dot" style={{ background: "var(--faint)" }} />
          weather-model environment
        </p>
      </section>

      <section>
        <h2>What this is, and is not</h2>
        <ul className="cn-notes">
          {summary.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
          <li>
            Satellite: {summary.meta.satellite}. Environment: {summary.meta.environment}.
          </li>
          <li>Not a warning product. Radar and a lightning network are the next inputs; neither has an open archive.</li>
        </ul>
      </section>
    </main>
  )
}

function DayChart({ detail, lead }: { detail: NowcastDay; lead: string }) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 960
  const padL = 58
  const padR = 16
  const h1 = 150
  const h2 = 210
  const gap = 34
  const H = h1 + h2 + gap + 44
  const n = detail.times.length
  const t0 = Date.parse(detail.times[0])
  const t1 = Date.parse(detail.times[n - 1])
  const x = (iso: string) => padL + ((Date.parse(iso) - t0) / Math.max(t1 - t0, 1)) * (W - padL - padR)

  const probs = useMemo(() => SETS.map((s) => detail.probability[s][lead]), [detail, lead])
  const pmax = useMemo(() => {
    let m = 0.3
    probs.forEach((arr) => arr.forEach((v) => v !== null && v > m && (m = v)))
    return Math.min(1, Math.ceil(m * 10) / 10 + 0.05)
  }, [probs])

  const cttLo = 195
  const cttHi = 300
  const yC = (v: number) => 8 + ((v - cttLo) / (cttHi - cttLo)) * (h1 - 16)
  const yP = (v: number) => h1 + gap + 8 + (1 - v / pmax) * (h2 - 16)

  const path = (values: (number | null)[], y: (v: number) => number) => {
    let d = ""
    let pen = false
    values.forEach((v, i) => {
      if (v === null) {
        pen = false
        return
      }
      d += `${pen ? "L" : "M"}${x(detail.times[i]).toFixed(1)},${y(v).toFixed(1)}`
      pen = true
    })
    return d
  }

  const truth = detail.truth[lead]
  const bands: { from: number; to: number }[] = []
  truth.forEach((v, i) => {
    if (v === 1) {
      const last = bands[bands.length - 1]
      if (last && last.to === i - 1) last.to = i
      else bands.push({ from: i, to: i })
    }
  })
  const step = n > 1 ? (x(detail.times[1]) - x(detail.times[0])) / 2 : 6

  const ticks: number[] = []
  detail.times.forEach((iso, i) => {
    const label = istLabel(iso)
    if (label.endsWith(":30") && Number(label.slice(0, 2)) % 3 === 2) ticks.push(i)
  })

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const px = ((event.clientX - rect.left) / rect.width) * W
    let best = 0
    let bestD = Infinity
    detail.times.forEach((iso, i) => {
      const dd = Math.abs(x(iso) - px)
      if (dd < bestD) {
        bestD = dd
        best = i
      }
    })
    setHover(best)
  }

  const hx = hover === null ? 0 : x(detail.times[hover])
  const tipRight = hover !== null && hx > W * 0.6

  return (
    <div className="cn-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cloud-top temperature and thunderstorm probability for the selected day" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {/* cloud-top panel */}
        <text className="cn-axis" x={padL - 8} y={14} textAnchor="end">
          colder
        </text>
        {[200, 235, 270].map((v) => (
          <g key={v}>
            <line className={v === 235 ? "cn-thresh" : "cn-grid"} x1={padL} x2={W - padR} y1={yC(v)} y2={yC(v)} />
            <text className="cn-axis" x={padL - 8} y={yC(v) + 4} textAnchor="end">
              {v} K
            </text>
          </g>
        ))}
        <text className="cn-axis" x={W - padR} y={yC(235) - 4} textAnchor="end">
          235 K · deep-cloud threshold
        </text>
        <path d={path(detail.series.ctt_min, yC)} className="cn-line" style={{ stroke: "var(--cyan)" }} />
        <text className="cn-panel" x={padL} y={h1 + 4}>
          Coldest cloud top within 1° (INSAT-3DS)
        </text>

        {/* probability panel */}
        {bands.map((b) => (
          <rect
            key={b.from}
            x={x(detail.times[b.from]) - step}
            width={x(detail.times[b.to]) - x(detail.times[b.from]) + step * 2}
            y={h1 + gap}
            height={h2}
            className="cn-band"
          />
        ))}
        {[0, pmax / 2, pmax].map((v) => (
          <g key={v}>
            <line className="cn-grid" x1={padL} x2={W - padR} y1={yP(v)} y2={yP(v)} />
            <text className="cn-axis" x={padL - 8} y={yP(v) + 4} textAnchor="end">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {SETS.map((s, i) => (
          <path key={s} d={path(probs[i], yP)} className="cn-line" style={{ stroke: COLOR[s], strokeWidth: s === "env+sat" ? 2.4 : 1.6, opacity: s === "env" ? 0.8 : 1 }} />
        ))}
        {detail.reports.map((r) => (
          <line key={r.time} x1={x(r.time)} x2={x(r.time)} y1={h1 + gap + h2 - 14} y2={h1 + gap + h2} className="cn-report" />
        ))}
        <text className="cn-panel" x={padL} y={h1 + gap - 8}>
          Probability of a thunderstorm at the airport within +{lead} min
        </text>

        {/* x axis */}
        {ticks.map((i) => (
          <text key={i} className="cn-axis" x={x(detail.times[i])} y={H - 18} textAnchor="middle">
            {istLabel(detail.times[i])}
          </text>
        ))}
        <text className="cn-axis" x={W - padR} y={H - 2} textAnchor="end">
          Time, IST · red ticks: storm reported at the airport · {istDate(detail.times[0])} 05:30 to next day 05:30
        </text>

        {hover !== null && (
          <g>
            <line x1={hx} x2={hx} y1={4} y2={h1 + gap + h2} className="cn-cursor" />
            <g transform={`translate(${tipRight ? hx - 218 : hx + 10}, ${h1 + gap + 14})`}>
              <rect width="208" height="104" rx="4" className="cn-tip" />
              <text x="10" y="18" className="cn-tip-head">
                {istLabel(detail.times[hover])} IST
              </text>
              <text x="10" y="36" className="cn-tip-text">
                Cloud top {detail.series.ctt_min[hover] === null ? "n/a (clear)" : `${num(detail.series.ctt_min[hover], 0)} K`}
              </text>
              {SETS.map((s, i) => (
                <text key={s} x="10" y={54 + i * 16} className="cn-tip-text" style={{ fill: COLOR[s] }}>
                  {SHORT[s]}: {pct(probs[i][hover], 0)}
                </text>
              ))}
              <text x="10" y="100" className="cn-tip-text">
                {truth[hover] === 1 ? "Storm followed" : "No storm followed"}
              </text>
            </g>
          </g>
        )}
      </svg>
      <div className="cn-legend">
        {SETS.map((s) => (
          <span key={s}>
            <span className="cn-dot" style={{ background: COLOR[s] }} />
            {SHORT[s]}
          </span>
        ))}
        <span>
          <span className="cn-dot cn-dot-band" />
          storm reported within +{lead} min
        </span>
      </div>
      {detail.reports.length > 0 && (
        <details className="cn-reports">
          <summary>{detail.reports.length} airport reports of thunderstorm that day (raw METAR)</summary>
          <ul>
            {detail.reports.map((r) => (
              <li key={r.time}>
                <span className="mono">{istLabel(r.time)} IST</span> {r.metar}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}


const LIVE_REFRESH_MS = 5 * 60 * 1000

function LiveCard({ city, cityName }: { city: string; cityName: string }) {
  const [live, setLive] = useState<LiveForecast | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancel = false
    setLoading(true)
    api
      .nowcastLive(city)
      .then((r) => {
        if (cancel) return
        setLive(r)
        setError(null)
      })
      .catch((reason: unknown) => !cancel && setError(reason instanceof Error ? reason.message : "Live data did not load."))
      .finally(() => !cancel && setLoading(false))
    return () => {
      cancel = true
    }
  }, [city, tick])

  useEffect(() => {
    const id = window.setInterval(() => setTick((n) => n + 1), LIVE_REFRESH_MS)
    return () => window.clearInterval(id)
  }, [])

  const d = live?.drivers
  const main = live?.models.env_sat_obs

  return (
    <section className="cn-live">
      <div className="cn-live-head">
        <h2>
          <span className={`cn-pulse ${live ? "on" : ""}`} aria-hidden="true" />
          Live now · {cityName}
        </h2>
        <button type="button" className="text-button" onClick={() => setTick((n) => n + 1)} disabled={loading}>
          {loading ? "Updating" : "Refresh"}
        </button>
      </div>
      {error && !live && <p className="fine">Live mode is unavailable right now: {error}</p>}
      {live && main && (
        <>
          <p className="fine">
            Newest INSAT-3DS scene {istLabel(live.scene_time)} IST ({live.scene_age_min} min ago). Probabilities count from that scene time. Experimental: a research
            prototype, not a warning.
          </p>
          {live.stale && <p className="fine">Showing the last good forecast; the newest data could not be fetched ({live.stale_reason}).</p>}
          <div className="cn-live-grid">
            {Object.entries(main.leads).map(([minute, r]) => {
              const sat = live.models.env_sat.leads[minute]
              return (
                <div key={minute} className={`cn-live-cell ${r.alert ? "alert" : ""}`}>
                  <span className="cn-card-label">Next {minute} min</span>
                  <span className="cn-card-value">{pct(r.probability)}</span>
                  <span className="cn-live-bar">
                    <span style={{ width: `${Math.min(100, r.probability * 100)}%` }} />
                    <i style={{ left: `${Math.min(100, r.threshold * 100)}%` }} title="alert threshold" />
                  </span>
                  <span className="cn-card-note">
                    {r.alert ? "Above the alert threshold" : "Below the alert threshold"} · usual rate {pct(r.base_rate)}
                  </span>
                  <span className="cn-card-note">Satellite + weather only: {pct(sat.probability)}</span>
                </div>
              )
            })}
          </div>
          <dl className="cn-live-facts">
            <div>
              <dt>Coldest cloud top</dt>
              <dd>{d?.coldest_cloud_top_k == null ? "clear" : `${num(d.coldest_cloud_top_k, 0)} K`}</dd>
            </div>
            <div>
              <dt>Change in 30 min</dt>
              <dd>{d?.cloud_top_change_30min_k == null ? "n/a" : `${d.cloud_top_change_30min_k > 0 ? "+" : ""}${num(d.cloud_top_change_30min_k, 1)} K`}</dd>
            </div>
            <div>
              <dt>CAPE</dt>
              <dd>{d?.cape == null ? "n/a" : `${num(d.cape, 0)} J/kg`}</dd>
            </div>
            <div>
              <dt>Lifted index</dt>
              <dd>{num(d?.lifted_index, 1)}</dd>
            </div>
            <div>
              <dt>Storm at airport, last hour</dt>
              <dd>{live.airport.storm_reported_last_hour ? "yes" : "no"}</dd>
            </div>
          </dl>
          {live.airport.storm_reports_since_scene.length > 0 && (
            <p className="fine">
              Thunderstorm reported at {live.station} since the scene: {live.airport.storm_reports_since_scene.map((r) => `${istLabel(r.time)} IST`).join(", ")}.
            </p>
          )}
          {live.airport.latest_report && (
            <p className="fine mono">
              Latest report {istLabel(live.airport.latest_report.time)} IST · {live.airport.latest_report.metar}
            </p>
          )}
        </>
      )}
      {!live && !error && <p className="quiet">Fetching the newest satellite scene.</p>}
    </section>
  )
}
