import { BrowserRouter, Route, Routes } from "react-router-dom"
import { TopBar } from "./components/Chrome"
import { AnalyticsPage } from "./pages/AnalyticsPage"
import { MapPage } from "./pages/MapPage"
import { MethodologyPage } from "./pages/MethodologyPage"
import { AppProvider } from "./state"

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <div className="shell">
          <TopBar />
          <Routes>
            <Route path="/" element={<MapPage />} />
            <Route path="/model" element={<AnalyticsPage />} />
            <Route path="/method" element={<MethodologyPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </AppProvider>
  )
}
