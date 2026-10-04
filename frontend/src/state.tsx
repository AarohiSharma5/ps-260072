import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

type Theme = "dark" | "light"

interface AppContextValue {
  theme: Theme
  setTheme: (theme: Theme) => void
}

const AppContext = createContext<AppContextValue | null>(null)

function initialTheme(): Theme {
  const stored = localStorage.getItem("akash-theme")
  return stored === "light" ? "light" : "dark"
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem("akash-theme", theme)
  }, [theme])

  const value = useMemo(() => ({ theme, setTheme }), [theme])
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const value = useContext(AppContext)
  if (!value) throw new Error("useApp must be used inside AppProvider")
  return value
}
