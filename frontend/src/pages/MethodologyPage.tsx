import { useEffect, useState } from "react"
import { api } from "../api"
import { Blocker } from "../components/Chrome"
import { useApp } from "../state"
import type { Methodology } from "../types"

export function MethodologyPage() {
  const { modelLoaded, loading, error, status } = useApp()
  const [doc, setDoc] = useState<Methodology | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    if (!modelLoaded) return
    let cancel = false
    api
      .methodology()
      .then((next) => {
        if (!cancel) setDoc(next)
      })
      .catch((reason: unknown) => {
        if (!cancel) setLoadError(reason instanceof Error ? reason.message : "Methodology unavailable")
      })
    return () => {
      cancel = true
    }
  }, [modelLoaded])

  if (loading || error || !modelLoaded) return <Blocker />
  if (loadError || !doc) {
    return (
      <main className="page">
        <h1>{loadError ?? "Reading the method note."}</h1>
      </main>
    )
  }

  const groups = new Map<string, Methodology["features"]>()
  doc.features.forEach((feature) => {
    const list = groups.get(feature.group) ?? []
    list.push(feature)
    groups.set(feature.group, list)
  })

  return (
    <main className="page method">
      <p className="kicker">SIH 26072 · {doc.data_mode}</p>
      <h1>{doc.title}</h1>
      <p className="lede">{doc.disclaimer}</p>
      {doc.source_name && <p className="quiet">Source: {doc.source_name}</p>}
      <p className="quiet">
        Time step {doc.step_min} min. Leads {doc.leads_min.map((m) => `+${m}`).join(", ")} min. Hazards trained: {doc.hazards.join(", ")}.
        {doc.skipped_targets.length > 0 && ` Skipped for too few events: ${doc.skipped_targets.join(", ")}.`}
      </p>
      <ol className="pipeline">
        {doc.steps.map((step, index) => (
          <li key={step.id}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <div>
              <h2>{step.title}</h2>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
      <section>
        <h2>Split and leakage</h2>
        <p>
          {doc.split.scheme}. {doc.split.train_cases} train cases ({doc.split.train_rows.toLocaleString()} rows),{" "}
          {doc.split.val_cases} validation cases ({doc.split.val_rows.toLocaleString()} rows),           {doc.split.test_cases} test cases (
          {doc.split.test_rows.toLocaleString()} rows). First training case {doc.split.first_train_case}; last test case {doc.split.last_test_case}.
        </p>
        <ul>
          {doc.split.leakage_controls.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
      <section>
        <h2>Features the trained model actually uses</h2>
        {[...groups.entries()].map(([group, features]) => (
          <div key={group} className="feature-group">
            <h3>{group}</h3>
            <ul>
              {features.map((feature) => (
                <li key={feature.id}>
                  {feature.label}
                  <span>{feature.unit}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>
      <section>
        <h2>Baselines</h2>
        <p>{doc.baselines.persistence}</p>
        <p>{doc.baselines.extrapolation}</p>
      </section>
      <section>
        <h2>Prototype risk thresholds</h2>
        <p>
          Watch {doc.thresholds.watch}, high {doc.thresholds.high}, severe {doc.thresholds.severe}. Status: {doc.thresholds.status}.
          These cuts are not an official warning standard.
        </p>
      </section>
      <section>
        <h2>Feeds</h2>
        <ul className="feed-list">
          {status?.datasets.map((dataset) => (
            <li key={dataset.id}>
              <span className={`status-dot status-${dataset.status.toLowerCase()}`}>{dataset.status}</span>
              <span>
                {dataset.label}. {dataset.note}
              </span>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h2>Scaling beyond this sector</h2>
        <ol>
          {doc.scaling.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ol>
      </section>
      <section>
        <h2>Limitations</h2>
        <ul>
          {doc.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  )
}
