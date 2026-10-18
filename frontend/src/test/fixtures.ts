import type { Job } from "../api/types";
import dev_004337 from "./fixtures/dev_004337.json";
import dev_004623 from "./fixtures/dev_004623.json";
import dev_005417 from "./fixtures/dev_005417.json";
import dev_005675 from "./fixtures/dev_005675.json";
import dev_034103 from "./fixtures/dev_034103.json";
import truth from "./fixtures/ground_truth.json";

/** Real API responses from scripts/44_frontend_fixtures.py (LAV-DF dev, shipped model). */
export const FIXTURES: Record<string, Job> = {
  dev_004337: dev_004337 as Job,
  dev_034103: dev_034103 as Job,
  dev_004623: dev_004623 as Job,
  dev_005675: dev_005675 as Job,
  dev_005417: dev_005417 as Job,
};

export const GROUND_TRUTH = truth as Record<
  string,
  { note: string; label: number; class_name: string; fake_periods: number[][] }
>;

export const REAL = FIXTURES.dev_004337;
export const PARTIAL = FIXTURES.dev_034103;
export const TWO_SPANS = FIXTURES.dev_004623;

export function failedJob(code: string): Job {
  return {
    ...REAL,
    status: "failed",
    result: null,
    error: { code, message: code, detail: null },
  };
}
