import { NavLink } from "react-router-dom"
import { useApp } from "../state"

export function TopBar() {
  const { theme, setTheme } = useApp()
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
          Nowcast
        </NavLink>
        <NavLink to="/model">Model</NavLink>
        <NavLink to="/method">Method</NavLink>
      </nav>
      <div className="top-meta">
        <span className="mode-pill mode-historical" title="INSAT-3DS cloud tops from MOSDAC, Open-Meteo environment, airport thunderstorm reports as truth. Cross-validated on 2025 data.">
          INSAT-3DS · 2025 CASES
        </span>
        <button type="button" className="text-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  )
}
