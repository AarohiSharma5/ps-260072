import { BrowserRouter, Route, Routes } from "react-router-dom"
import { TopBar } from "./components/Chrome"
import { MethodPage } from "./pages/MethodPage"
import { ModelPage } from "./pages/ModelPage"
import { NowcastPage } from "./pages/NowcastPage"
import { AppProvider } from "./state"

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <div className="shell">
          <TopBar />
          <Routes>
            <Route path="/" element={<NowcastPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/method" element={<MethodPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </AppProvider>
  )
}
