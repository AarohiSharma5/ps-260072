import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { api } from "./api"
import type { CaseSummary, Status } from "./types"

type Theme = "dark" | "light"

interface AppContextValue {
  theme: Theme
  setTheme: (theme: Theme) => void
  status: Status | null
  cases: CaseSummary[]
  caseId: string | null
  setCaseId: (id: string) => void
  loading: boolean
  error: string | null
  modelLoaded: boolean
}

const AppContext = createContext<AppContextValue | null>(null)

function initialTheme(): Theme {
  const stored = localStorage.getItem("akash-theme")
  return stored === "light" ? "light" : "dark"
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(initialTheme)
  const [status, setStatus] = useState<Status | null>(null)
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [caseId, setCaseId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem("akash-theme", theme)
  }, [theme])

  useEffect(() => {
    let cancel = false
    api
      .health()
      .then(async (health) => {
        if (cancel) return
        if (!health.model_loaded) {
          setModelLoaded(false)
          setError("Model artifacts are missing. Train the prototype, then start the API.")
          return
        }
        const [nextStatus, nextCases] = await Promise.all([api.status(), api.cases()])
        if (cancel) return
        setStatus(nextStatus)
        setCases(nextCases.cases)
        setCaseId(nextCases.cases[0]?.id ?? null)
        setModelLoaded(true)
      })
      .catch((reason: unknown) => {
        if (cancel) return
        const message = reason instanceof Error ? reason.message : "The API did not respond."
        setError(message)
      })
      .finally(() => {
        if (!cancel) setLoading(false)
      })
    return () => {
      cancel = true
    }
  }, [])

  const value = useMemo<AppContextValue>(
    () => ({
      theme,
      setTheme: setThemeState,
      status,
      cases,
      caseId,
      setCaseId,
      loading,
      error,
      modelLoaded,
    }),
    [theme, status, cases, caseId, loading, error, modelLoaded],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const value = useContext(AppContext)
  if (!value) throw new Error("useApp must be used inside AppProvider")
  return value
}
