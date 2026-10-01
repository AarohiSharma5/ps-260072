interface TimelineProps {
  offsets: number[]
  offset: number
  playing: boolean
  clock: string
  onOffset: (offset: number) => void
  onPlaying: (playing: boolean) => void
}

export function Timeline({ offsets, offset, playing, clock, onOffset, onPlaying }: TimelineProps) {
  const index = Math.max(0, offsets.indexOf(offset))
  return (
    <footer className="timebar">
      <button type="button" className="play" onClick={() => onPlaying(!playing)} aria-pressed={playing}>
        {playing ? "Pause" : "Play"}
      </button>
      <div className="time-readout">
        <strong>{offset === 0 ? "NOW" : `+${offset} MIN`}</strong>
        <span>{offset === 0 ? `Analysis ${clock}` : `Valid ${clock}`}</span>
      </div>
      <div className="time-track">
        <input
          type="range"
          min={0}
          max={offsets.length - 1}
          step={1}
          value={index}
          aria-label="Forecast lead"
          onChange={(event) => {
            onPlaying(false)
            onOffset(offsets[Number(event.target.value)] ?? 0)
          }}
        />
        <div className="ticks">
          {offsets.map((minute) => (
            <button
              key={minute}
              type="button"
              className={minute === offset ? "tick on" : "tick"}
              onClick={() => {
                onPlaying(false)
                onOffset(minute)
              }}
            >
              {minute === 0 ? "NOW" : `+${minute}`}
            </button>
          ))}
        </div>
      </div>
    </footer>
  )
}
