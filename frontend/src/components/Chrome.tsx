import { NavLink } from "react-router-dom"
import { clockStamp } from "../format"
import { useApp } from "../state"

export function TopBar() {
  const { theme, setTheme, status, cases, caseId, setCaseId } = useApp()
  return (
    <header className="topbar">
      <div className="brand">
        <svg viewBox="0 0 16 16" className="mark" aria-hidden="true">
          <path d="M9 1 L4 9 H8 L7 15 L13 6 H9 Z" />
        </svg>
        <div>
          <div className="brand-name">AKASH</div>
          <div className="brand-sub">Thunderstorm and lightning nowcast</div>
        </div>
      </div>
      <nav className="nav">
        <NavLink to="/" end>
          Map
        </NavLink>
        <NavLink to="/model">Model</NavLink>
        <NavLink to="/method">Method</NavLink>
      </nav>
      <div className="top-meta">
        {cases.length > 0 && (
          <label className="case-select">
            <span>Case</span>
            <select value={caseId ?? ""} onChange={(event) => setCaseId(event.target.value)}>
              {cases.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.title} · {item.clock}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className={`mode-pill mode-${(status?.data_mode ?? "demo").toLowerCase()}`} title={status?.disclaimer ?? ""}>
          {status?.data_mode === "HISTORICAL" ? "ERA5 HISTORICAL" : status?.data_mode === "SEVIR" ? "SEVIR STORMS" : "DEMO DATA"}
        </span>
        <span className="meta-block">
          <span className="meta-label">Model</span>
          <span>{status?.model_version ?? "not loaded"}</span>
        </span>
        <span className="meta-block">
          <span className="meta-label">Trained</span>
          <span>{clockStamp(status?.model_trained_at)}</span>
        </span>
        <button
          type="button"
          className="text-button"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  )
}

export function Blocker() {
  const { loading, error, modelLoaded } = useApp()
  if (loading) {
    return (
      <div className="blocker">
        <p className="kicker">AKASH</p>
        <h1>Loading the sector.</h1>
      </div>
    )
  }
  if (!modelLoaded || error) {
    return (
      <div className="blocker">
        <p className="kicker">Prototype offline</p>
        <h1>The nowcast API is not ready.</h1>
        <p>{error ?? "Start the backend to load the trained model."}</p>
        <pre>{`cd backend
source .venv/bin/activate
python -m app.train                       # simulator (DEMO)
# or, with a CDS key in ~/.cdsapirc:
python -m app.era5 all && python -m app.train   # ERA5 (HISTORICAL)
python -m uvicorn app.main:app --port 8000`}</pre>
      </div>
    )
  }
  return null
}
