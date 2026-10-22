import { pitchLines } from "./geometry";

const TEAM_FILL: Record<string, string> = { A: "#f1f5f9", B: "#3b82f6", other: "#fbbf24", unknown: "#94a3b8" };

function PitchBase({ L, W, children }: { L: number; W: number; children?: React.ReactNode }) {
  const pad = 3;
  return (
    <svg className="pitch-svg" viewBox={`${-pad} ${-pad} ${L + 2 * pad} ${W + 2 * pad}`} preserveAspectRatio="xMidYMid meet">
      <rect x={-pad} y={-pad} width={L + 2 * pad} height={W + 2 * pad} className="pitch-bg" />
      {Array.from({ length: Math.ceil(L / 10) }, (_, k) => (
        k % 2 === 0 ? <rect key={k} x={k * 10} y={0} width={Math.min(10, L - k * 10)} height={W} className="pitch-stripe" /> : null
      ))}
      {pitchLines(L, W).map((p, i) => (
        <polyline key={i} points={p.map((q) => q.join(",")).join(" ")} className="pitch-line" />
      ))}
      {children}
    </svg>
  );
}

export function MiniPitch({ L, W, points, colors, next }: { L: number; W: number; points: number[][]; colors: string[]; next: number }) {
  return (
    <div className="mini-pitch">
      <PitchBase L={L} W={W}>
        <polygon points={points.map((p) => p.join(",")).join(" ")} className="calib-region" />
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p[0]} cy={p[1]} r={i === next ? 2.6 : 1.6} fill={colors[i]} className={i === next ? "pulse" : ""} />
            <text x={p[0] + 2} y={p[1] - 2} fontSize={4} fill={colors[i]} fontWeight={700}>P{i + 1}</text>
          </g>
        ))}
      </PitchBase>
    </div>
  );
}

export type FramePlayers = { frame: number; t: number; status: string; players: { id: number; team: string; x?: number; y?: number }[] };

export function TopDown({
  L, W, frame, trails, showTrails, heat, heatCell, calibQuad,
}: {
  L: number; W: number; frame: FramePlayers | null; trails: { id: number; team: string; points: number[][] }[];
  showTrails: boolean; heat: number[][] | null; heatCell: number; calibQuad?: number[][];
}) {
  const hmax = heat ? Math.max(1e-6, ...heat.flat()) : 1;
  return (
    <PitchBase L={L} W={W}>
      {heat && heat.map((row, r) => row.map((v, c) => v > 0 && (
        <rect key={`${r}-${c}`} x={c * heatCell} y={r * heatCell} width={heatCell} height={heatCell}
          fill="#f97316" opacity={0.08 + 0.72 * (v / hmax)} />
      )))}
      {calibQuad && <polygon points={calibQuad.map((p) => p.join(",")).join(" ")} className="calib-region thin" />}
      {showTrails && trails.map((t) => (
        <polyline key={t.id} points={t.points.map((p) => p.join(",")).join(" ")} fill="none"
          stroke={TEAM_FILL[t.team] ?? "#94a3b8"} strokeOpacity={0.45} strokeWidth={0.35} />
      ))}
      {frame?.players.filter((p) => p.x !== undefined).map((p) => (
        <g key={p.id}>
          <circle cx={p.x} cy={p.y} r={1.25} fill={TEAM_FILL[p.team] ?? "#94a3b8"} stroke="#0b1220" strokeWidth={0.3} />
          <text x={p.x! + 1.5} y={p.y! - 1.2} fontSize={2.2} fill="#cbd5e1">{p.id}</text>
        </g>
      ))}
    </PitchBase>
  );
}

export const teamColor = (t: string) => TEAM_FILL[t] ?? "#94a3b8";
