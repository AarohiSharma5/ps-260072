import { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { api } from "./api"
import type { NowcastCity, NowcastDomain, NowcastSummary } from "./types"

/** Cities, their map footprints and the selected city (from ?city=). */
export function useCities() {
  const [params, setParams] = useSearchParams()
  const [cities, setCities] = useState<NowcastCity[]>([])
  const [domains, setDomains] = useState<NowcastDomain[]>([])
  const [defaultCity, setDefaultCity] = useState("chennai")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .nowcastCities()
      .then((r) => {
        setCities(r.cities)
        setDefaultCity(r.default)
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "The API did not respond."))
    api
      .nowcastDomains()
      .then((r) => setDomains(r.domains))
      .catch(() => setDomains([]))
  }, [])

  const requested = params.get("city")
  const ready = cities.filter((c) => c.ready)
  const city = ready.find((c) => c.key === requested)?.key ?? ready.find((c) => c.key === defaultCity)?.key ?? ready[0]?.key ?? null
  const select = (key: string) => setParams({ city: key })
  return { cities, ready, domains, city, select, error }
}

export function useSummary(city: string | null) {
  const [summary, setSummary] = useState<NowcastSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!city) return
    let cancel = false
    setSummary(null)
    setError(null)
    api
      .nowcastSummary(city)
      .then((s) => !cancel && setSummary(s))
      .catch((reason: unknown) => !cancel && setError(reason instanceof Error ? reason.message : "Results did not load."))
    return () => {
      cancel = true
    }
  }, [city])
  return { summary, error }
}
