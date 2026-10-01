import { useEffect, useRef, useState, type MutableRefObject } from "react"
import maplibregl, { type GeoJSONSource, type Map, type StyleSpecification } from "maplibre-gl"
import type { FeatureCollection } from "geojson"
import "maplibre-gl/dist/maplibre-gl.css"
import { buildCollections } from "../mapdata"
import type { BaseLayer, Hazard, Overlays, Scenario } from "../types"
import { coordinates, formatNumber } from "../format"

interface MapCanvasProps {
  scenario: Scenario
  base: BaseLayer
  hazard: Hazard
  offset: number
  overlays: Overlays
  opacity: number
  theme: "dark" | "light"
  selected: number | null
  detailOpen: boolean
  onSelect: (id: number) => void
}

const DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
const LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

function fallbackStyle(theme: "dark" | "light"): StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [
      {
        id: "background",
        type: "background",
        paint: { "background-color": theme === "dark" ? "#0c1216" : "#d9d3c8" },
      },
    ],
  }
}

export function MapCanvas({
  scenario,
  base,
  hazard,
  offset,
  overlays,
  opacity,
  theme,
  selected,
  detailOpen,
  onSelect,
}: MapCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<Map | null>(null)
  const onSelectRef = useRef(onSelect)
  const [ready, setReady] = useState(0)
  const [hover, setHover] = useState<{ x: number; y: number; id: number; value: string } | null>(null)
  onSelectRef.current = onSelect

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let cancelled = false
    const map = new maplibregl.Map({
      container,
      style: fallbackStyle(theme),
      center: [scenario.domain.center_lon, scenario.domain.center_lat],
      zoom: 8,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      fadeDuration: 0,
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: false }), "top-right")
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left")
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right")
    map.on("style.load", () => {
      ensureLayers(map)
      bind(map, setHover, onSelectRef)
      if (!cancelled) setReady((value) => value + 1)
    })
    const remote = theme === "dark" ? DARK_STYLE : LIGHT_STYLE
    fetch(remote)
      .then((response) => {
        if (!response.ok || cancelled) return
        map.setStyle(remote)
      })
      .catch(() => {
        /* Local fallback style remains. */
      })
    mapRef.current = map
    return () => {
      cancelled = true
      map.remove()
      mapRef.current = null
    }
  }, [theme, scenario.domain.center_lat, scenario.domain.center_lon])

  useEffect(() => {
    const map = mapRef.current
    if (!map || ready === 0 || !map.getSource("cells")) return
    const data = buildCollections(scenario, base, hazard, offset, overlays)
    ;(map.getSource("cells") as GeoJSONSource).setData(data.cells as FeatureCollection)
    ;(map.getSource("wind") as GeoJSONSource).setData(data.wind as FeatureCollection)
    ;(map.getSource("motion") as GeoJSONSource).setData(data.motion as FeatureCollection)
    ;(map.getSource("lightning") as GeoJSONSource).setData(data.lightning as FeatureCollection)
    ;(map.getSource("verify-flashes") as GeoJSONSource).setData(data.verifyFlashes as FeatureCollection)
    ;(map.getSource("graticule") as GeoJSONSource).setData(data.graticule as FeatureCollection)
    ;(map.getSource("domain") as GeoJSONSource).setData(data.domain as FeatureCollection)
    map.setPaintProperty("cell-fill", "fill-opacity", ["*", ["get", "fade"], opacity])
    map.setPaintProperty("cell-line", "line-opacity", overlays.grid ? 0.45 : 0)
    map.setPaintProperty("cell-verify", "line-opacity", overlays.verify ? 0.95 : 0)
    map.setFilter("cell-select", ["==", ["get", "id"], selected ?? -1])
  }, [ready, scenario, base, hazard, offset, overlays, opacity, selected])

  useEffect(() => {
    const map = mapRef.current
    if (!map || ready === 0) return
    map.fitBounds(
      [
        [scenario.domain.lon_min, scenario.domain.lat_min],
        [scenario.domain.lon_max, scenario.domain.lat_max],
      ],
      { padding: { top: 36, bottom: 128, left: 348, right: 56 }, duration: 0 },
    )
  }, [ready, scenario.domain.lat_min, scenario.domain.lon_min, scenario.domain.lat_max, scenario.domain.lon_max, scenario.case.id])

  useEffect(() => {
    const map = mapRef.current
    if (!map || ready === 0) return
    const markers: maplibregl.Marker[] = []
    if (overlays.places) {
      scenario.places.forEach((place) => {
        const element = document.createElement("div")
        element.className = "place"
        element.textContent = place.name
        markers.push(new maplibregl.Marker({ element, anchor: "bottom" }).setLngLat([place.lon, place.lat]).addTo(map))
      })
    }
    return () => markers.forEach((marker) => marker.remove())
  }, [ready, overlays.places, scenario.places, scenario.case.id])

  const hovered = hover ? scenario.grid[hover.id] : null
  return (
    <div className={detailOpen ? "map-wrap detail-open" : "map-wrap"}>
      <div ref={containerRef} className="map" />
      {hovered && hover && (
        <div className="hover-card" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <div className="hover-kicker">Grid {hovered.grid_id}</div>
          <div>{coordinates(hovered.latitude, hovered.longitude)}</div>
          <div className="hover-value">{hover.value === "" ? "Unavailable" : formatNumber(Number(hover.value), 2)}</div>
        </div>
      )}
    </div>
  )
}

function bind(
  map: Map,
  setHover: (hover: { x: number; y: number; id: number; value: string } | null) => void,
  onSelectRef: MutableRefObject<(id: number) => void>,
) {
  const flagged = map as Map & { __akashBound?: boolean }
  if (flagged.__akashBound) return
  flagged.__akashBound = true
  map.on("click", "cell-fill", (event) => {
    const feature = event.features?.[0]
    if (!feature) return
    onSelectRef.current(Number(feature.properties?.id))
  })
  map.on("mousemove", "cell-fill", (event) => {
    map.getCanvas().style.cursor = "pointer"
    const feature = event.features?.[0]
    if (!feature) return
    setHover({
      x: event.point.x,
      y: event.point.y,
      id: Number(feature.properties?.id),
      value: String(feature.properties?.value ?? ""),
    })
  })
  map.on("mouseleave", "cell-fill", () => {
    map.getCanvas().style.cursor = ""
    setHover(null)
  })
}

function ensureLayers(map: Map) {
  if (map.getSource("cells")) return
  map.addSource("graticule", { type: "geojson", data: emptyCollection() })
  map.addSource("domain", { type: "geojson", data: emptyCollection() })
  map.addSource("cells", { type: "geojson", data: emptyCollection() })
  map.addSource("wind", { type: "geojson", data: emptyCollection() })
  map.addSource("motion", { type: "geojson", data: emptyCollection() })
  map.addSource("lightning", { type: "geojson", data: emptyCollection() })
  map.addSource("verify-flashes", { type: "geojson", data: emptyCollection() })

  map.addLayer({
    id: "graticule",
    type: "line",
    source: "graticule",
    paint: { "line-color": "#8d887c", "line-width": 0.6, "line-opacity": 0.25 },
  })
  map.addLayer({
    id: "domain-line",
    type: "line",
    source: "domain",
    paint: { "line-color": "#e4b15c", "line-width": 1.4, "line-opacity": 0.8 },
  })
  map.addLayer({
    id: "cell-fill",
    type: "fill",
    source: "cells",
    paint: { "fill-color": ["get", "color"], "fill-opacity": ["get", "fade"] },
  })
  map.addLayer({
    id: "cell-line",
    type: "line",
    source: "cells",
    paint: { "line-color": "#efeae2", "line-width": 0.6, "line-opacity": 0.35 },
  })
  map.addLayer({
    id: "cell-verify",
    type: "line",
    source: "cells",
    filter: ["==", ["get", "verify"], 1],
    paint: { "line-color": "#f4f1ea", "line-width": 1.8, "line-dasharray": [1.2, 0.8], "line-opacity": 0 },
  })
  map.addLayer({
    id: "cell-select",
    type: "line",
    source: "cells",
    filter: ["==", ["get", "id"], -1],
    paint: { "line-color": "#ffffff", "line-width": 2.4 },
  })
  map.addLayer({
    id: "wind-line",
    type: "line",
    source: "wind",
    paint: { "line-color": "#d5e6ef", "line-width": 1.3, "line-opacity": 0.85 },
  })
  map.addLayer({
    id: "motion-line",
    type: "line",
    source: "motion",
    paint: { "line-color": "#e4b15c", "line-width": 2.1, "line-opacity": 0.95 },
  })
  map.addLayer({
    id: "lightning-circle",
    type: "circle",
    source: "lightning",
    paint: {
      "circle-radius": ["get", "radius"],
      "circle-color": "#9be7ff",
      "circle-opacity": ["get", "fade"],
      "circle-stroke-color": "#f4fbff",
      "circle-stroke-width": 0.7,
    },
  })
  map.addLayer({
    id: "verify-circle",
    type: "circle",
    source: "verify-flashes",
    paint: {
      "circle-radius": 7,
      "circle-color": "rgba(0,0,0,0)",
      "circle-stroke-color": "#f7f4ee",
      "circle-stroke-width": 1.6,
    },
  })
}

function emptyCollection() {
  return { type: "FeatureCollection" as const, features: [] }
}
