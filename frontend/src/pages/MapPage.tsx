import { useEffect, useState } from "react"
import { api } from "../api"
import { Blocker } from "../components/Chrome"
import { DetailPanel } from "../components/DetailPanel"
import { LayerPanel } from "../components/LayerPanel"
import { MapCanvas } from "../components/MapCanvas"
import { Timeline } from "../components/Timeline"
import { legendFor, rampGradient } from "../mapdata"
import { useApp } from "../state"
import type { BaseLayer, Hazard, Overlays, Scenario } from "../types"

const INITIAL_OVERLAYS: Overlays = {
  grid: true,
  lightning: true,
  wind: false,
  motion: true,
  verify: false,
  places: true,
}

export function MapPage() {
  const { caseId, theme, modelLoaded, loading, error } = useApp()
  const [scenario, setScenario] = useState<Scenario | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [base, setBase] = useState<BaseLayer>("reflectivity")
  const [hazard, setHazard] = useState<Hazard>("thunderstorm")
  const [offset, setOffset] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [overlays, setOverlays] = useState<Overlays>(INITIAL_OVERLAYS)
  const [opacity, setOpacity] = useState(0.84)
  const [selected, setSelected] = useState<number | null>(null)

  useEffect(() => {
    if (!caseId || !modelLoaded) return
    let cancel = false
    setLoadError(null)
    api
      .scenario(caseId)
      .then((next) => {
        if (cancel) return
        setScenario(next)
        setOffset(0)
        setSelected(null)
        setPlaying(false)
        setBase((current) => {
          const layer = next.layers.find((item) => item.id === current)
          if (layer && !layer.available) return next.analysis.vil ? "vil" : next.analysis.convective_precip ? "convective_precip" : "cape"
          if (current === "lightning" && !next.hazards.includes("lightning")) return "thunderstorm"
          if (current === "extrapolation" && next.extrapolation === null) return "thunderstorm"
          return current
        })
      })
      .catch((reason: unknown) => {
        if (!cancel) setLoadError(reason instanceof Error ? reason.message : "Could not load the case.")
      })
    return () => {
      cancel = true
    }
  }, [caseId, modelLoaded])

  useEffect(() => {
    if (!playing || !scenario) return
    const timer = window.setInterval(() => {
      setOffset((current) => {
        const timeline = scenario.timeline_min
        const index = timeline.indexOf(current)
        return timeline[(index + 1) % timeline.length] ?? 0
      })
    }, 1100)
    return () => window.clearInterval(timer)
  }, [playing, scenario])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) return
      if (!scenario) return
      if (event.code === "Space") {
        event.preventDefault()
        setPlaying((value) => !value)
      }
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault()
        setPlaying(false)
        setOffset((current) => {
          const timeline = scenario.timeline_min
          const index = Math.max(0, timeline.indexOf(current))
          const next = event.key === "ArrowRight" ? index + 1 : index - 1
          return timeline[Math.min(timeline.length - 1, Math.max(0, next))] ?? current
        })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [scenario])

  if (loading || error || !modelLoaded) return <Blocker />
  if (loadError) {
    return (
      <div className="blocker">
        <h1>This case did not load.</h1>
        <p>{loadError}</p>
      </div>
    )
  }
  if (!scenario) {
    return (
      <div className="blocker">
        <p className="kicker">DEMO</p>
        <h1>Preparing the sector.</h1>
      </div>
    )
  }

  const legend = legendFor(scenario, base, hazard, offset, overlays)
  const clockIndex = scenario.verification.offsets_min.indexOf(offset)
  const clock = clockIndex >= 0 ? scenario.verification.clocks[clockIndex] : scenario.analysis.clock
  const firstLead = scenario.panel_leads_min[0] ?? scenario.leads_min[0]
  const analysisBase: BaseLayer = scenario.analysis.reflectivity ? "reflectivity" : scenario.analysis.vil ? "vil" : scenario.analysis.convective_precip ? "convective_precip" : "cape"

  return (
    <div className="stage">
      <MapCanvas
        scenario={scenario}
        base={base}
        hazard={hazard}
        offset={offset}
        overlays={overlays}
        opacity={opacity}
        theme={theme}
        selected={selected}
        detailOpen={selected !== null}
        onSelect={setSelected}
      />
      <LayerPanel
        scenario={scenario}
        base={base}
        hazard={hazard}
        offset={offset}
        overlays={overlays}
        opacity={opacity}
        onBase={setBase}
        onHazard={setHazard}
        onOverlays={setOverlays}
        onOpacity={setOpacity}
        onPreset={(preset) => {
          setPlaying(false)
          if (preset === "analysis") {
            setBase(analysisBase)
            setOffset(0)
            setOverlays((current) => ({ ...current, lightning: true, wind: !scenario.analysis.reflectivity, verify: false }))
          } else {
            setBase("thunderstorm")
            setHazard("thunderstorm")
            setOffset(firstLead)
            setOverlays((current) => ({ ...current, lightning: true, motion: true, verify: false }))
          }
        }}
      />
      {selected !== null && (
        <DetailPanel
          scenario={scenario}
          gridId={selected}
          base={base}
          hazard={hazard}
          offset={offset}
          onClose={() => setSelected(null)}
        />
      )}
      <aside className="legend">
        <div className="legend-title">
          <span>{legend.title}</span>
          <span>{legend.unit}</span>
        </div>
        <div className="ramp" style={{ background: rampGradient(legend.stops) }} />
        <div className="ramp-labels">
          <span>{legend.minLabel}</span>
          <span>{legend.maxLabel}</span>
        </div>
        <p>{legend.caption}</p>
        {overlays.verify && <p>White dashed cells are the thunderstorm label at this valid time ({scenario.data_mode}).</p>}
        {overlays.motion && scenario.analysis.reflectivity && <p>Amber arrows extend estimated storm motion by the selected lead.</p>}
        {overlays.lightning && scenario.hazards.includes("lightning") && offset > 0 && <p>Cyan marks are model lightning probability at or above the watch cut, not strike points.</p>}
      </aside>
      <Timeline
        offsets={scenario.timeline_min}
        offset={offset}
        playing={playing}
        clock={clock}
        onOffset={setOffset}
        onPlaying={(next) => {
          if (next && base !== "thunderstorm" && base !== "lightning" && base !== "extrapolation") {
            setBase("thunderstorm")
            setHazard("thunderstorm")
            setOverlays((current) => ({ ...current, motion: true, lightning: true }))
          }
          setPlaying(next)
        }}
      />
    </div>
  )
}
