"""Prototype configuration.

Two data modes share one pipeline:

DEMO        deterministic simulator on a 15-minute step (radar, satellite,
            lightning, environment all synthetic).
HISTORICAL  ERA5 hourly single-level reanalysis converted to NPZ cases in
            backend/data/historical. ERA5 carries the environment (T, Td,
            p, 10 m wind, CAPE, CIN, K index, Totals-Totals) and a
            convective-precipitation proxy label. It has no radar, satellite
            brightness temperature, or lightning; those stay UNAVAILABLE.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
DATA_DIR = ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
ERA5_RAW_DIR = DATA_DIR / "era5_raw"
SEVIR_DIR = DATA_DIR / "sevir"
SEVIR_META_DIR = DATA_DIR / "sevir_meta"

MODEL_VERSION = "akash-xgb-0.4.0"

# Simulator sector. Cell size is about 9 km.
LAT_MIN = 20.40
LAT_MAX = 22.00
LON_MIN = 78.20
LON_MAX = 80.00
N_LAT = 20
N_LON = 20
DOMAIN_NAME = "Nagpur sector"

N_TIMES = 16
DT_MIN = 15

# ERA5 defaults. Native 0.25 degree grid. Central India, pre-monsoon season.
ERA5_AREA = {"north": 25.0, "west": 74.0, "south": 17.0, "east": 84.0}
ERA5_DOMAIN_NAME = "Central India (ERA5 0.25°)"
ERA5_DEFAULT_START = "2025-04-01"
ERA5_DEFAULT_END = "2025-05-31"

# Prototype event definitions. Not IMD warning criteria.
THUNDERSTORM_DBZ = 40.0
LIGHTNING_COUNT_MIN = 1.0
CONVECTIVE_PRECIP_MM_H = 1.0  # ERA5 proxy label threshold
VIL_KG_M2 = 5.0  # SEVIR vertically integrated liquid proxy threshold

# Prototype risk thresholds applied to predicted probability.
RISK_WATCH = 0.35
RISK_HIGH = 0.60
RISK_SEVERE = 0.80
RISK_THRESHOLD_STATUS = "prototype"

# Simulator split. Demo seeds are never included here.
TRAIN_SEEDS = tuple(range(1100, 1124))
VAL_SEEDS = tuple(range(1124, 1130))
TEST_SEEDS = tuple(range(1130, 1136))
DEMO_SEEDS = (9001, 9002, 9003)

REFERENCE_PLACES = (
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
    {"name": "Wardha", "lat": 20.7453, "lon": 78.6022},
    {"name": "Bhandara", "lat": 21.1667, "lon": 79.6500},
    {"name": "Kamptee", "lat": 21.2230, "lon": 79.2020},
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    {"name": "Jabalpur", "lat": 23.1815, "lon": 79.9864},
    {"name": "Raipur", "lat": 21.2514, "lon": 81.6296},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    {"name": "Aurangabad", "lat": 19.8762, "lon": 75.3433},
    {"name": "Indore", "lat": 22.7196, "lon": 75.8577},
    {"name": "Amravati", "lat": 20.9320, "lon": 77.7523},
    {"name": "Nanded", "lat": 19.1383, "lon": 77.3210},
)

ALL_HAZARDS = ("thunderstorm", "lightning")


def leads_for_step(dt_min: int) -> tuple[int, ...]:
    """Forecast leads that the time step can actually support."""
    if dt_min <= 15:
        return (15, 30, 45, 60, 90)
    if dt_min == 30:
        return (30, 60, 90)
    return (dt_min, 2 * dt_min, 3 * dt_min)


def panel_leads_for_step(dt_min: int) -> tuple[int, ...]:
    leads = leads_for_step(dt_min)
    if len(leads) <= 3:
        return leads
    return tuple(lead for lead in (30, 60, 90) if lead in leads)


# Labels use {s1} and {s2} for one and two time steps.
FEATURE_META: dict[str, dict[str, str]] = {
    "reflectivity": {"label": "Radar reflectivity", "unit": "dBZ", "group": "radar"},
    "refl_trend_1": {"label": "Reflectivity change ({s1})", "unit": "dBZ / {s1}", "group": "radar"},
    "refl_trend_2": {"label": "Reflectivity change ({s2})", "unit": "dBZ / {s2}", "group": "radar"},
    "refl_grad_mag": {"label": "Reflectivity spatial gradient", "unit": "dBZ / km", "group": "radar"},
    "ir_bt": {"label": "Infrared brightness temperature", "unit": "K", "group": "satellite"},
    "ctt_cooling_1": {"label": "Cloud-top cooling ({s1})", "unit": "K / {s1}", "group": "satellite"},
    "ctt_cooling_2": {"label": "Cloud-top cooling ({s2})", "unit": "K / {s2}", "group": "satellite"},
    "ir_grad_mag": {"label": "Brightness-temperature gradient", "unit": "K / km", "group": "satellite"},
    "temperature": {"label": "Temperature", "unit": "°C", "group": "atmosphere"},
    "temp_tendency_1": {"label": "Temperature tendency", "unit": "°C / {s1}", "group": "atmosphere"},
    "rh": {"label": "Relative humidity", "unit": "%", "group": "atmosphere"},
    "rh_tendency_1": {"label": "Humidity tendency", "unit": "pp / {s1}", "group": "atmosphere"},
    "pressure": {"label": "Pressure", "unit": "hPa", "group": "atmosphere"},
    "pressure_tendency_1": {"label": "Pressure tendency", "unit": "hPa / {s1}", "group": "atmosphere"},
    "wind_speed": {"label": "Wind speed", "unit": "m/s", "group": "atmosphere"},
    "wind_sin": {"label": "Wind direction (sine)", "unit": "1", "group": "atmosphere"},
    "wind_cos": {"label": "Wind direction (cosine)", "unit": "1", "group": "atmosphere"},
    "convergence": {"label": "Wind convergence", "unit": "10⁻⁵ s⁻¹", "group": "atmosphere"},
    "cape": {"label": "CAPE", "unit": "J/kg", "group": "instability"},
    "cape_tendency_1": {"label": "CAPE tendency", "unit": "J/kg / {s1}", "group": "instability"},
    "cape_grad_mag": {"label": "CAPE spatial gradient", "unit": "J/kg / km", "group": "instability"},
    "cin": {"label": "CIN", "unit": "J/kg", "group": "instability"},
    "shear": {"label": "Bulk shear magnitude", "unit": "m/s", "group": "instability"},
    "k_index": {"label": "K index", "unit": "°C", "group": "instability"},
    "total_totals": {"label": "Totals-Totals index", "unit": "°C", "group": "instability"},
    "convective_precip": {"label": "Convective precipitation", "unit": "mm/h", "group": "precipitation"},
    "cp_trend_1": {"label": "Convective precipitation change ({s1})", "unit": "mm/h / {s1}", "group": "precipitation"},
    "neighbor_cp_max": {"label": "Neighborhood maximum convective precipitation", "unit": "mm/h", "group": "precipitation"},
    "lightning_count": {"label": "Lightning count", "unit": "flashes / {s1}", "group": "lightning"},
    "lightning_density": {"label": "Lightning density", "unit": "flashes / km² / {s1}", "group": "lightning"},
    "lightning_rate_change": {"label": "Lightning-rate change", "unit": "flashes / {s1}", "group": "lightning"},
    "neighbor_lightning": {"label": "Neighborhood lightning count", "unit": "flashes / {s1}", "group": "lightning"},
    "motion_u": {"label": "Storm motion eastward", "unit": "m/s", "group": "motion"},
    "motion_v": {"label": "Storm motion northward", "unit": "m/s", "group": "motion"},
    "motion_speed": {"label": "Storm motion speed", "unit": "m/s", "group": "motion"},
    "upstream_refl_1": {"label": "Upstream reflectivity ({s1} motion)", "unit": "dBZ", "group": "motion"},
    "neighbor_refl_max": {"label": "Neighborhood maximum reflectivity", "unit": "dBZ", "group": "radar"},
    "vil": {"label": "Vertically integrated liquid", "unit": "kg/m²", "group": "radar"},
    "vil_trend_1": {"label": "VIL change ({s1})", "unit": "kg/m² / {s1}", "group": "radar"},
    "vil_trend_2": {"label": "VIL change ({s2})", "unit": "kg/m² / {s2}", "group": "radar"},
    "vil_grad_mag": {"label": "VIL spatial gradient", "unit": "kg/m² / km", "group": "radar"},
    "neighbor_vil_max": {"label": "Neighborhood maximum VIL", "unit": "kg/m²", "group": "radar"},
    "upstream_vil_1": {"label": "Upstream VIL ({s1} motion)", "unit": "kg/m²", "group": "motion"},
}

FEATURE_REQUIRES: dict[str, tuple[str, ...]] = {
    "reflectivity": ("reflectivity",),
    "refl_trend_1": ("reflectivity",),
    "refl_trend_2": ("reflectivity",),
    "refl_grad_mag": ("reflectivity",),
    "neighbor_refl_max": ("reflectivity",),
    "motion_u": ("reflectivity",),
    "motion_v": ("reflectivity",),
    "motion_speed": ("reflectivity",),
    "upstream_refl_1": ("reflectivity",),
    "vil": ("vil",),
    "vil_trend_1": ("vil",),
    "vil_trend_2": ("vil",),
    "vil_grad_mag": ("vil",),
    "neighbor_vil_max": ("vil",),
    "upstream_vil_1": ("vil",),
    "ir_bt": ("ir_bt",),
    "ctt_cooling_1": ("ir_bt",),
    "ctt_cooling_2": ("ir_bt",),
    "ir_grad_mag": ("ir_bt",),
    "temperature": ("temperature",),
    "temp_tendency_1": ("temperature",),
    "rh": ("rh",),
    "rh_tendency_1": ("rh",),
    "pressure": ("pressure",),
    "pressure_tendency_1": ("pressure",),
    "wind_speed": ("wind_u", "wind_v"),
    "wind_sin": ("wind_u", "wind_v"),
    "wind_cos": ("wind_u", "wind_v"),
    "convergence": ("wind_u", "wind_v"),
    "cape": ("cape",),
    "cape_tendency_1": ("cape",),
    "cape_grad_mag": ("cape",),
    "cin": ("cin",),
    "shear": ("shear",),
    "k_index": ("k_index",),
    "total_totals": ("total_totals",),
    "convective_precip": ("convective_precip",),
    "cp_trend_1": ("convective_precip",),
    "neighbor_cp_max": ("convective_precip",),
    "lightning_count": ("lightning_count",),
    "lightning_density": ("lightning_count",),
    "lightning_rate_change": ("lightning_count",),
    "neighbor_lightning": ("lightning_count",),
}

OBSERVATION_FIELDS = (
    "reflectivity",
    "ir_bt",
    "temperature",
    "rh",
    "pressure",
    "wind_u",
    "wind_v",
    "cape",
    "cin",
    "shear",
    "k_index",
    "total_totals",
    "convective_precip",
    "lightning_count",
    "vil",
)


def step_text(minutes: int) -> str:
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} h"
    return f"{minutes} min"


def feature_meta(dt_min: int) -> dict[str, dict[str, str]]:
    s1 = step_text(dt_min)
    s2 = step_text(2 * dt_min)
    out: dict[str, dict[str, str]] = {}
    for name, meta in FEATURE_META.items():
        out[name] = {
            "label": meta["label"].format(s1=s1, s2=s2),
            "unit": meta["unit"].format(s1=s1, s2=s2),
            "group": meta["group"],
        }
    return out


DEMO_DATASETS = (
    {"id": "demo_radar", "label": "Radar reflectivity", "status": "DEMO",
     "note": "Synthetic gridded reflectivity. No radar is connected."},
    {"id": "demo_satellite", "label": "Infrared brightness temperature", "status": "DEMO",
     "note": "Synthetic brightness temperature, smoother than the reflectivity field."},
    {"id": "demo_atmosphere", "label": "Temperature, humidity, pressure, wind", "status": "DEMO",
     "note": "Synthetic near-storm environment. Not an observation or an NWP analysis."},
    {"id": "demo_instability", "label": "CAPE, CIN, bulk shear", "status": "DEMO",
     "note": "Synthetic instability fields. Shear is a single magnitude."},
    {"id": "demo_lightning", "label": "Lightning counts", "status": "DEMO",
     "note": "Synthetic flashes on the prototype grid. Not a lightning-network feed."},
    {"id": "imd_radar", "label": "IMD Doppler radar mosaic", "status": "UNAVAILABLE",
     "note": "No live or archive radar feed is connected."},
    {"id": "insat", "label": "INSAT infrared imagery", "status": "UNAVAILABLE",
     "note": "No satellite API is connected."},
    {"id": "lightning_network", "label": "Operational lightning network", "status": "UNAVAILABLE",
     "note": "No lightning feed is connected."},
    {"id": "era5", "label": "ERA5 reanalysis", "status": "UNAVAILABLE",
     "note": "Run python -m app.era5 with a CDS API key to switch to historical mode."},
)

ERA5_DATASETS = (
    {"id": "era5_environment", "label": "ERA5 2 m temperature, dewpoint, pressure, 10 m wind", "status": "HISTORICAL",
     "note": "Copernicus ERA5 hourly single-level reanalysis, 0.25°, read from the public ARCO-ERA5 zarr (same dataset as the CDS, DOI 10.24381/cds.adbb2d47). Reanalysis, not station observations."},
    {"id": "era5_instability", "label": "ERA5 CAPE, CIN, K index, Totals-Totals", "status": "HISTORICAL",
     "note": "IFS-derived instability from the same single-level reanalysis."},
    {"id": "era5_cp", "label": "ERA5 convective precipitation (proxy label)", "status": "HISTORICAL",
     "note": "Hourly convective precipitation from the reanalysis. The thunderstorm label is this field at or above the proxy threshold. It is not an observed storm report."},
    {"id": "era5_pressure", "label": "ERA5 pressure-level wind (bulk shear)", "status": "UNAVAILABLE",
     "note": "The pressure-level dataset (DOI 10.24381/cds.bd0915c6) would supply 850 hPa and 500 hPa wind for shear. The CDS copy needs an API key. The public ARCO copy stores all 37 levels in one global chunk per hour, which this prototype does not download. Shear is left absent."},
    {"id": "imd_radar", "label": "IMD Doppler radar", "status": "UNAVAILABLE",
     "note": "mausam.imd.gov.in serves a live radar viewer, not a gridded archive for the seeded days. Reflectivity was not reconstructed from images."},
    {"id": "insat", "label": "INSAT-3D / INSAT-3DS brightness temperature", "status": "UNAVAILABLE",
     "note": "MOSDAC L1/L2 products require an account. Public GSICS files are calibration coefficients, not brightness-temperature grids, so no infrared field was seeded."},
    {"id": "gpm_imerg", "label": "GPM IMERG precipitation", "status": "UNAVAILABLE",
     "note": "Half-hourly IMERG on the GPM PPS servers requires a registered email. It was not downloaded, and ERA5 convective precipitation was not replaced with it."},
    {"id": "lis_vhrmc", "label": "LIS very-high-resolution lightning climatology", "status": "UNAVAILABLE",
     "note": "LIS VHRMC (DOI 10.5067/LIS/LIS/DATA302) is a 1998–2013 monthly flash-rate climatology and needs an Earthdata login. It is not an hourly lightning observation, so it is not used as a lightning label."},
)

DEMO_DISCLAIMER = (
    "DEMO MODE. Every meteorological value is simulated. Predictions come from an XGBoost model "
    "trained on that simulator. They are not live forecasts and must not be used for warnings."
)

SEVIR_DATASETS = (
    {"id": "sevir_vil", "label": "SEVIR vertically integrated liquid", "status": "SEVIR",
     "note": "NEXRAD VIL on a United States storm patch, coarsened from 1 km to 8 km by an 8×8 block average. Not reflectivity in dBZ, and not an Indian radar."},
    {"id": "sevir_lightning", "label": "SEVIR lightning counts", "status": "SEVIR",
     "note": "Flash counts from the SEVIR lightning grid, summed onto the same 8 km cells. Geostationary Lightning Mapper style counts for that US patch."},
    {"id": "sevir_ir", "label": "SEVIR infrared brightness temperature", "status": "UNAVAILABLE",
     "note": "ir069 and ir107 files are in the Drive archive but were not aligned into these cases. Cooling features stay off."},
    {"id": "mumbai_analogue", "label": "Mumbai-analogue selection", "status": "SEVIR",
     "note": "The event list is the Mumbai-analogue catalog. The images themselves are US storms chosen as analogues. The map is the US patch, not Mumbai."},
    {"id": "imd_radar", "label": "IMD Doppler radar", "status": "UNAVAILABLE",
     "note": "Not in this archive."},
    {"id": "insat", "label": "INSAT brightness temperature", "status": "UNAVAILABLE",
     "note": "Not in this archive."},
    {"id": "era5", "label": "ERA5 environment", "status": "UNAVAILABLE",
     "note": "These cases do not include temperature, CAPE, CIN, or wind analyses."},
)

SEVIR_DISCLAIMER = (
    "SEVIR MODE. Each case is a United States storm patch from the SEVIR dataset, chosen by the Mumbai-analogue catalog. "
    "It is not an observation over Mumbai or India. "
    "The radar field is vertically integrated liquid in kg/m², coarsened to 8 km. "
    "The thunderstorm label is VIL at or above 5 kg/m², a proxy, not 40 dBZ. "
    "Predictions are a research prototype, not warnings."
)

ERA5_DISCLAIMER = (
    "HISTORICAL MODE. Fields are Copernicus ERA5 hourly single-level reanalysis for the seeded days, "
    "read from the public ARCO-ERA5 copy of the CDS dataset. "
    "The thunderstorm label is ERA5 convective precipitation at or above 1 mm/h, a reanalysis proxy, not an observed storm. "
    "IMD radar, INSAT brightness temperature, GPM IMERG, LIS lightning, and pressure-level shear were not seeded. "
    "Predictions are a research prototype, not warnings."
)
