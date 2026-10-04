import { DomainMap } from "../components/DomainMap"
import { useCities, useSummary } from "../hooks"

const STEPS: { title: string; body: string }[] = [
  {
    title: "Satellite: INSAT-3DS cloud tops",
    body: "MOSDAC product 3SIMG_L2B_CTP gives cloud-top temperature and pressure every 30 minutes at about 35 km pixel spacing. For each airport we take the 4° × 4° window around it and compute the coldest cloud top within 0.5°, 1° and 2°, the mean, the share of pixels colder than 235 K and 220 K, the coldest cloud top in each quadrant, and how much the cloud tops cooled over the last 30 and 60 minutes. Clear sky is kept: no cloud is information too.",
  },
  {
    title: "Environment: Open-Meteo model fields",
    body: "CAPE, convective inhibition, lifted index, the 850-500 hPa lapse rate, wind shear (surface to 850 hPa and 850 to 500 hPa), humidity at 700 hPa, cloud cover, pressure and temperature, with 3-hour tendencies. Training uses Open-Meteo's historical forecasts; the live card uses its current forecast. These are model data, not observations.",
  },
  {
    title: "Truth: what the airport reported",
    body: "A scene is a positive example if the airport's METAR or SPECI reports a thunderstorm (TS or VCTS in the present-weather group, ignoring TEMPO, BECMG, NOSIG and remarks) within the next 30, 60 or 90 minutes. Reports come from the Iowa Environmental Mesonet archive. This is not a lightning-network record.",
  },
  {
    title: "Model: XGBoost, one per lead time",
    body: "Three input sets are compared for each lead: weather model alone, plus the satellite, plus the airport's own thunderstorm reports of the last 3 hours. The third is the strongest but only helps once a storm is near; the second is the early-warning case.",
  },
  {
    title: "Validation: days the model never saw",
    body: "Five-fold cross-validation with whole 4-day blocks held out, so neighbouring scenes never leak. Inside each training fold, a 4-fold inner loop chooses between four regularised settings by log-loss and picks the alert threshold that maximises the critical success index. Baselines are time-of-day climatology and persistence (a storm in the last hour).",
  },
  {
    title: "Live: the same code on the newest scene",
    body: "The live card downloads the newest INSAT scene and the two before it, reads the current Open-Meteo forecast and the airport's latest reports, builds a row with exactly the same feature code as training, and scores it with the saved models. A new scene arrives every 30 minutes.",
  },
]

const LIMITS = [
  "One year of data (2025) and one airport per model. Skill on other years or places is untested.",
  "Rare events: a few hundred storm cases per city, so scores carry real uncertainty.",
  "A single airport report is a point observation. A storm 10 km away goes unreported, so some false alerts are not false.",
  "The 35 km satellite pixel cannot resolve individual cells. Radar and a lightning network would sharpen this; neither has an open archive.",
  "Live mode uses the weather forecast as the environment, which is not identical to the historical forecasts used for training.",
  "A research prototype, not a warning product.",
]

export function MethodPage() {
  const { ready, domains, city, select } = useCities()
  const { summary } = useSummary(city)

  return (
    <main className="page method">
      <p className="kicker">SIH 26072 · AIML thunderstorm nowcasting</p>
      <h1>How the nowcast is built.</h1>
      <p className="lede">
        A model that turns ISRO's INSAT-3DS cloud tops and a weather-model environment into the probability that an airport reports a thunderstorm within the next 30, 60 or 90 minutes.
      </p>

      {ready.length > 1 && city && (
        <div className="switch-row">
          <div className="switch" role="group" aria-label="City">
            {ready.map((c) => (
              <button key={c.key} type="button" className={c.key === city ? "on" : ""} onClick={() => select(c.key)}>
                {c.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <section>
        <h2>Where it is trained</h2>
        <DomainMap domains={domains} selected={city ?? ""} onSelect={select} />
        {summary && (
          <p className="quiet">
            {summary.meta.city_name}: {summary.meta.days} days, {summary.meta.scenes.toLocaleString()} satellite scenes, {summary.meta.season}.
          </p>
        )}
      </section>

      <ol className="pipeline">
        {STEPS.map((step, index) => (
          <li key={step.title}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <div>
              <h2>{step.title}</h2>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <section>
        <h2>Data sources</h2>
        <ul className="feed-list">
          <li>
            <span className="status-dot status-historical">LIVE + ARCHIVE</span>
            <span>INSAT-3DS 3SIMG_L2B_CTP, MOSDAC (ISRO). Cloud-top temperature and pressure.</span>
          </li>
          <li>
            <span className="status-dot status-historical">LIVE + ARCHIVE</span>
            <span>Open-Meteo forecast and historical-forecast APIs (CC-BY 4.0). Convective environment.</span>
          </li>
          <li>
            <span className="status-dot status-historical">LIVE + ARCHIVE</span>
            <span>METAR and SPECI from the Iowa Environmental Mesonet. Thunderstorm truth and recent-storm input.</span>
          </li>
          <li>
            <span className="status-dot status-unavailable">NOT USED</span>
            <span>IMD Doppler radar and a lightning network. No open archive was available to this prototype.</span>
          </li>
        </ul>
      </section>

      <section>
        <h2>Scaling beyond two cities</h2>
        <ol>
          <li>One MOSDAC download covers all of India; adding a city is one entry in the city list plus its airport reports and weather.</li>
          <li>Pool several cities and years into one model, and add the airport-free variant everywhere.</li>
          <li>Add radar reflectivity and lightning flashes as inputs when a feed is available, and score against them.</li>
        </ol>
      </section>

      <section>
        <h2>Limitations</h2>
        <ul>
          {LIMITS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  )
}
