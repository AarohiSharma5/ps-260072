import { useEffect, useRef, useState } from "react"
import maplibregl, { type GeoJSONSource, type Map, type StyleSpecification } from "maplibre-gl"
import type { FeatureCollection } from "geojson"
import "maplibre-gl/dist/maplibre-gl.css"
import type { NowcastDomain } from "../types"
import { useApp } from "../state"

const DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
const LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

function fallbackStyle(theme: "dark" | "light"): StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [{ id: "background", type: "background", paint: { "background-color": theme === "dark" ? "#0c1216" : "#d9d3c8" } }],
  }
}

function collections(domains: NowcastDomain[], selected: string) {
  const windows: FeatureCollection = {
    type: "FeatureCollection",
    features: domains.map((d) => {
      const [w, s, e, n] = d.window
      return {
        type: "Feature",
        properties: { key: d.city, ready: d.ready ? 1 : 0, selected: d.city === selected ? 1 : 0 },
        geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
      }
    }),
  }
  const pixels: FeatureCollection = {
    type: "FeatureCollection",
    features: domains.flatMap((d) =>
      d.pixels.map(([lon, lat]) => ({
        type: "Feature" as const,
        properties: { key: d.city, ready: d.ready ? 1 : 0, selected: d.city === selected ? 1 : 0 },
        geometry: { type: "Point" as const, coordinates: [lon, lat] },
      })),
    ),
  }
  return { windows, pixels }
}

function addLayers(map: Map) {
  const empty: FeatureCollection = { type: "FeatureCollection", features: [] }
  map.addSource("windows", { type: "geojson", data: empty })
  map.addSource("pixels", { type: "geojson", data: empty })
  map.addLayer({
    id: "window-fill",
    type: "fill",
    source: "windows",
    paint: { "fill-color": "#9be7ff", "fill-opacity": ["case", ["==", ["get", "selected"], 1], 0.12, 0.05] },
  })
  map.addLayer({
    id: "window-line",
    type: "line",
    source: "windows",
    paint: {
      "line-color": ["case", ["==", ["get", "selected"], 1], "#e4b15c", "#9be7ff"],
      "line-width": ["case", ["==", ["get", "selected"], 1], 2.4, 1.4],
      "line-dasharray": [3, 2],
    },
  })
  map.addLayer({
    id: "pixel-dots",
    type: "circle",
    source: "pixels",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 1.2, 6, 3, 8, 6],
      "circle-color": ["case", ["==", ["get", "selected"], 1], "#e4b15c", "#9be7ff"],
      "circle-opacity": ["case", ["==", ["get", "ready"], 1], 0.85, 0.4],
    },
  })
}

interface Props {
  domains: NowcastDomain[]
  selected: string
  onSelect?: (city: string) => void
  height?: number
}

/** Where the model was trained: INSAT-3DS pixels inside the 4x4 degree window around each airport. */
export function DomainMap({ domains, selected, onSelect, height = 380 }: Props) {
  const { theme } = useApp()
  const ref = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<Map | null>(null)
  const markers = useRef<maplibregl.Marker[]>([])
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect
  const [ready, setReady] = useState(0)

  useEffect(() => {
    const container = ref.current
    if (!container || domains.length === 0) return
    let cancelled = false
    const map = new maplibregl.Map({
      container,
      style: fallbackStyle(theme),
      center: [80, 20],
      zoom: 3.5,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      fadeDuration: 0,
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false, visualizePitch: false }), "top-right")
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-left")
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right")
    map.on("style.load", () => {
      addLayers(map)
      if (!cancelled) setReady((n) => n + 1)
    })
    const remote = theme === "dark" ? DARK_STYLE : LIGHT_STYLE
    fetch(remote)
      .then((r) => {
        if (r.ok && !cancelled) map.setStyle(remote)
      })
      .catch(() => {
        /* offline: the plain background and the pixel footprint still draw */
      })
    mapRef.current = map
    return () => {
      cancelled = true
      markers.current.forEach((m) => m.remove())
      markers.current = []
      map.remove()
      mapRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, domains.length])

  useEffect(() => {
    const map = mapRef.current
    if (!map || ready === 0 || !map.getSource("windows")) return
    const data = collections(domains, selected)
    ;(map.getSource("windows") as GeoJSONSource).setData(data.windows)
    ;(map.getSource("pixels") as GeoJSONSource).setData(data.pixels)

    markers.current.forEach((m) => m.remove())
    markers.current = domains.map((d) => {
      const el = document.createElement("button")
      el.type = "button"
      el.className = `dm-pin ${d.city === selected ? "on" : ""} ${d.ready ? "" : "pending"}`
      el.innerHTML = `<b>${d.name}</b><span>${d.station}${d.ready ? "" : " · training"}</span>`
      el.onclick = () => selectRef.current?.(d.city)
      return new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([d.lon, d.lat]).addTo(map)
    })

    const lons = domains.flatMap((d) => [d.window[0], d.window[2]])
    const lats = domains.flatMap((d) => [d.window[1], d.window[3]])
    map.fitBounds(
      [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
      { padding: 48, duration: 0, maxZoom: 6 },
    )
  }, [ready, domains, selected])

  if (domains.length === 0) return null
  return (
    <figure className="dm">
      <div ref={ref} className="dm-canvas" style={{ height }} role="img" aria-label="Map of the INSAT-3DS pixels each model was trained on" />
      <figcaption className="fine">
        Dashed boxes are the 4° × 4° training windows. Dots are the INSAT-3DS cloud-top pixels (about 35 km apart) that feed each model; the label marks the airport
        whose thunderstorm reports are the truth.
      </figcaption>
    </figure>
  )
}
