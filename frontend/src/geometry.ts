export type Mat3 = number[][];

export function inv3(m: Mat3): Mat3 {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

export function apply(m: Mat3, x: number, y: number): [number, number, number] {
  const u = m[0][0] * x + m[0][1] * y + m[0][2];
  const v = m[1][0] * x + m[1][1] * y + m[1][2];
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  return [u / w, v / w, w];
}

/** Pitch template polylines in metres (same template as the backend renderer). */
export function pitchLines(L = 105, W = 68): number[][][] {
  const rect = (x0: number, y0: number, x1: number, y1: number) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
  const arc = (cx: number, cy: number, r: number, a0: number, a1: number, n = 48) =>
    Array.from({ length: n + 1 }, (_, k) => {
      const a = ((a0 + ((a1 - a0) * k) / n) * Math.PI) / 180;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    });
  const c = W / 2;
  const out = [rect(0, 0, L, W), [[L / 2, 0], [L / 2, W]], arc(L / 2, c, 9.15, 0, 360, 96)];
  for (const [gx, s] of [[0, 1], [L, -1]] as const) {
    out.push(rect(gx, c - 20.16, gx + s * 16.5, c + 20.16));
    out.push(rect(gx, c - 9.16, gx + s * 5.5, c + 9.16));
    out.push(s > 0 ? arc(gx + 11, c, 9.15, -53, 53) : arc(gx - 11, c, 9.15, 127, 233));
  }
  return out.map(densify);
}

function densify(p: number[][]): number[][] {
  const out = [p[0]];
  for (let k = 1; k < p.length; k++) {
    const [ax, ay] = p[k - 1], [bx, by] = p[k];
    const n = Math.max(1, Math.round(Math.hypot(bx - ax, by - ay)));
    for (let t = 1; t <= n; t++) out.push([ax + ((bx - ax) * t) / n, ay + ((by - ay) * t) / n]);
  }
  return out;
}

/** Project the pitch template into the image with the inverse of an image->pitch homography. */
export function projectTemplate(H: Mat3, L: number, W: number, imgW: number, imgH: number): number[][][] {
  const Hi = inv3(H);
  const runs: number[][][] = [];
  const wRef = Math.sign(apply(Hi, L / 2, W)[2]) || 1;
  for (const poly of pitchLines(L, W)) {
    let run: number[][] = [];
    for (const [x, y] of poly) {
      const [u, v, w] = apply(Hi, x, y);
      const ok = Math.sign(w) === wRef && u > -imgW && u < 2 * imgW && v > -imgH && v < 2 * imgH;
      if (ok) run.push([u, v]);
      else {
        if (run.length > 1) runs.push(run);
        run = [];
      }
    }
    if (run.length > 1) runs.push(run);
  }
  return runs;
}
