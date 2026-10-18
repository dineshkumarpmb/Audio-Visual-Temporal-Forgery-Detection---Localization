import { useEffect, useRef, useState } from "react";

import { ApiError, getJob } from "../api/client";
import type { Job } from "../api/types";

export const POLL_INTERVAL_MS = 1000;
/** Consecutive failed polls before giving up: a blip is retried, a dead server is reported. */
export const MAX_POLL_FAILURES = 5;

interface PollState {
  job: Job | null;
  error: ApiError | null;
  /** True while polls are failing but we have not given up yet. */
  reconnecting: boolean;
}

/** Poll `GET /api/v1/jobs/{id}` every second until the job finishes (P14-3). */
export function useJobPolling(jobId: string | null, fetchJob = getJob): PollState {
  const [state, setState] = useState<PollState>({ job: null, error: null, reconnecting: false });
  const fetchRef = useRef(fetchJob);
  fetchRef.current = fetchJob;

  useEffect(() => {
    setState({ job: null, error: null, reconnecting: false });
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const tick = async () => {
      try {
        const job = await fetchRef.current(jobId);
        if (cancelled) return;
        failures = 0;
        setState({ job, error: null, reconnecting: false });
        if (job.status === "succeeded" || job.status === "failed") return;
      } catch (e) {
        if (cancelled) return;
        const err = e instanceof ApiError ? e : new ApiError(0, "unexpected_response", String(e));
        failures += 1;
        // A 404 is final (the job expired or the server restarted); anything else may pass.
        if (err.status === 404 || failures >= MAX_POLL_FAILURES) {
          setState((s) => ({ ...s, error: err, reconnecting: false }));
          return;
        }
        setState((s) => ({ ...s, reconnecting: true }));
      }
      timer = setTimeout(tick, POLL_INTERVAL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return state;
}
