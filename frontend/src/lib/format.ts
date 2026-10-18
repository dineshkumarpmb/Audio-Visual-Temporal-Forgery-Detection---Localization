/** Seconds as m:ss.s — short clips need the tenth; LAV-DF spans are often under a second. */
export function formatTime(s: number): string {
  const sign = s < 0 ? "-" : "";
  const abs = Math.abs(s);
  const m = Math.floor(abs / 60);
  const rest = abs - m * 60;
  return `${sign}${m}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function formatDuration(s: number): string {
  return s < 10 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`;
}

export function formatPercent(p: number, digits = 0): string {
  const v = p * 100;
  // Never round a non-certain score up to 100% or down to 0%: a model is never certain.
  if (v > 99.9 && v < 100) return ">99.9%";
  if (v < 0.1 && v > 0) return "<0.1%";
  return `${v.toFixed(digits)}%`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function extensionOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
}
