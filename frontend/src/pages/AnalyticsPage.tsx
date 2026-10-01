import { useEffect, useState } from "react"
import { api } from "../api"
import { Blocker } from "../components/Chrome"
import { metricText } from "../format"
import { useApp } from "../state"
import type { Hazard, MetricsReport, OperatingMetrics } from "../types"

export function AnalyticsPage() {
  const { modelLoaded, loading, error, cases, caseId } = useApp()
  const [report, setReport] = useState<MetricsReport | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [hazard, setHazard] = useState<Hazard>("thunderstorm")
  const [lead, setLead] = useState<number | null>(null)

  useEffect(() => {
    if (!modelLoaded) return
    let cancel = false
    api
      .metrics()
      .then((next) => {
        if (cancel) return
        setReport(next)
        setLead(next.leads_min.includes(60) ? 60 : next.leads_min[0])
      })
      .catch((reason: unknown) => !cancel && setLoadError(reason instanceof Error ? reason.message : "Metrics unavailable"))
    return () => {
      cancel = true
    }
  }, [modelLoaded])

  if (loading || error || !modelLoaded) return <Blocker />
  if (loadError) {
    return (
      <main className="page">
        <h1>Metrics did not load.</h1>
        <p>{loadError}</p>
      </main>
    )
  }
  if (!report || lead === null) {
    return (
      <main className="page">
        <h1>Reading the test block.</h1>
      </main>
    )
  }

  const key = `${hazard}_${lead}`
  const target = report.targets[key]
  const demo = cases.find((item) => item.id === caseId)?.scores?.targets[key]

  return (
    <main className="page">
      <p className="kicker">Model verification · {report.data_mode}</p>
      <h1>Test-set performance</h1>
      <p className="lede">{report.disclaimer}</p>
      {report.data_mode === "HISTORICAL" && <p className="quiet">Label: {report.label_definitions.thunderstorm}</p>}
      <p className="quiet">{report.independence_note}</p>
      <p className="quiet">{report.threshold_note}</p>
      <dl className="facts inline">
        <dt>Version</dt>
        <dd>{report.model_version}</dd>
        <dt>Trained</dt>
        <dd>{report.trained_at}</dd>
        <dt>Step</dt>
        <dd>{report.step_min} min</dd>
      </dl>

      <div className="switch-row">
        <div className="switch">
          {(["thunderstorm", "lightning"] as Hazard[]).map((item) => (
            <button key={item} type="button" disabled={!report.hazards.includes(item)} className={item === hazard ? "on" : ""} onClick={() => setHazard(item)}>
              {item}
              {!report.hazards.includes(item) && " · unavailable"}
            </button>
          ))}
        </div>
        <div className="switch">
          {report.leads_min.map((minute) => (
            <button key={minute} type="button" className={minute === lead ? "on" : ""} onClick={() => setLead(minute)}>
              +{minute}
            </button>
          ))}
        </div>
      </div>

      {!target && <p>No metrics were stored for {key}. Either the hazard has no data in this mode or the target had too few positive events to train.</p>}
      {target && (
        <>
          <p className="quiet">
            Test rows {target.support.n.toLocaleString()}. Positive labels {target.support.positives.toLocaleString()} (
            {target.support.positive_rate === null ? "Unavailable" : `${(target.support.positive_rate * 100).toFixed(1)}%`}). Trees {target.n_trees}.
          </p>
          <Comparison model={target.model} persistence={target.persistence} extrapolation={target.extrapolation} />
          {!target.extrapolation && <p className="fine">Motion extrapolation baseline is unavailable: it needs radar reflectivity, which this dataset does not have.</p>}
          <div className="report-grid">
            <section>
              <h2>Confusion matrix</h2>
              <p className="fine">Model at probability 0.5. Rows are the prototype label.</p>
              <Matrix confusion={target.model.confusion} />
            </section>
            <section>
              <h2>Predicted against observed</h2>
              <p className="fine">Reliability of the model probability on the test rows. The diagonal would be perfect calibration.</p>
              <Reliability bins={target.reliability} />
            </section>
          </div>
          <section>
            <h2>Feature importance</h2>
            <p className="fine">Share of XGBoost split gain for this lead. Gain is not a causal effect.</p>
            <Importance rows={target.feature_importance} />
          </section>
          <section>
            <h2>This case</h2>
            <p className="fine">{cases.find((item) => item.id === caseId)?.scores?.note}</p>
            {demo ? (
              <dl className="facts">
                <dt>Mean predicted probability</dt>
                <dd>{metricText(demo.mean_predicted)}</dd>
                <dt>Observed label fraction</dt>
                <dd>{metricText(demo.observed_fraction)}</dd>
                <dt>Brier score</dt>
                <dd>{metricText(demo.brier)}</dd>
                <dt>F1 at 0.5</dt>
                <dd>{metricText(demo.f1)}</dd>
                <dt>ROC-AUC</dt>
                <dd>{metricText(demo.roc_auc)}</dd>
              </dl>
            ) : (
              <p>Case scores are unavailable.</p>
            )}
            <p className="quiet">Cell-by-cell comparison for this case is the verification overlay on the map: white dashed outlines are the label at the selected valid time.</p>
          </section>
          <section>
            <h2>Skill across leads</h2>
            <LeadTable report={report} hazard={hazard} />
          </section>
        </>
      )}
    </main>
  )
}

function Comparison({ model, persistence, extrapolation }: { model: OperatingMetrics; persistence: OperatingMetrics; extrapolation: OperatingMetrics | null }) {
  const rows: { label: string; key: keyof OperatingMetrics }[] = [
    { label: "Precision", key: "precision" },
    { label: "Recall", key: "recall" },
    { label: "F1", key: "f1" },
    { label: "ROC-AUC", key: "roc_auc" },
    { label: "PR-AUC", key: "average_precision" },
    { label: "Brier", key: "brier" },
  ]
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Score</th>
          <th>XGBoost</th>
          <th>Persistence</th>
          <th>Extrapolation</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <th>{row.label}</th>
            <td>{metricText(num(model[row.key]))}</td>
            <td>{metricText(num(persistence[row.key]))}</td>
            <td>{extrapolation ? metricText(num(extrapolation[row.key])) : "Unavailable"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function num(value: OperatingMetrics[keyof OperatingMetrics]): number | null {
  return typeof value === "number" ? value : null
}

function Matrix({ confusion }: { confusion: OperatingMetrics["confusion"] }) {
  return (
    <table className="matrix">
      <thead>
        <tr>
          <th />
          <th>Predicted no</th>
          <th>Predicted yes</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Observed no</th>
          <td>{confusion.tn.toLocaleString()}</td>
          <td>{confusion.fp.toLocaleString()}</td>
        </tr>
        <tr>
          <th>Observed yes</th>
          <td>{confusion.fn.toLocaleString()}</td>
          <td>{confusion.tp.toLocaleString()}</td>
        </tr>
      </tbody>
    </table>
  )
}

function Reliability({ bins }: { bins: MetricsReport["targets"][string]["reliability"] }) {
  const width = 460
  const height = 280
  const pad = 36
  const plot = (value: number) => pad + value * (width - pad * 2)
  const plotY = (value: number) => height - pad - value * (height - pad * 2)
  const points = bins.filter((bin) => bin.count > 0 && bin.mean_predicted !== null && bin.fraction_positive !== null)
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="reliability" role="img" aria-label="Reliability diagram">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={pad} className="guide" />
      {points.map((bin) => (
        <circle key={`${bin.bin_lo}-${bin.bin_hi}`} cx={plot(bin.mean_predicted ?? 0)} cy={plotY(bin.fraction_positive ?? 0)} r={Math.max(3, Math.min(10, Math.sqrt(bin.count) / 8))} />
      ))}
      <text x={width / 2} y={height - 8} textAnchor="middle">
        Mean predicted probability
      </text>
      <text x={14} y={height / 2} transform={`rotate(-90 14 ${height / 2})`} textAnchor="middle">
        Observed fraction
      </text>
    </svg>
  )
}

function Importance({ rows }: { rows: MetricsReport["targets"][string]["feature_importance"] }) {
  const top = rows.slice(0, 12)
  const max = top[0]?.gain_share || 1
  return (
    <ol className="importance">
      {top.map((row) => (
        <li key={row.feature}>
          <span>{row.label}</span>
          <span className="bar">
            <span style={{ width: `${(row.gain_share / max) * 100}%` }} />
          </span>
          <span className="mono">{(row.gain_share * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ol>
  )
}

function LeadTable({ report, hazard }: { report: MetricsReport; hazard: Hazard }) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Lead</th>
          <th>Model F1</th>
          <th>Persistence F1</th>
          <th>Extrapolation F1</th>
          <th>Model ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        {report.leads_min.map((minute) => {
          const row = report.targets[`${hazard}_${minute}`]
          if (!row) {
            return (
              <tr key={minute}>
                <th>+{minute}</th>
                <td colSpan={4}>Not trained</td>
              </tr>
            )
          }
          return (
            <tr key={minute}>
              <th>+{minute}</th>
              <td>{metricText(row.model.f1)}</td>
              <td>{metricText(row.persistence.f1)}</td>
              <td>{row.extrapolation ? metricText(row.extrapolation.f1) : "Unavailable"}</td>
              <td>{metricText(row.model.roc_auc)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
