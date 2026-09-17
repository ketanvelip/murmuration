/**
 * Per-wave objectives.
 *
 * Two of the three are about where the flock *is* rather than how many birds
 * you have knocked out of it, which is the whole reason the collider grew a
 * census probe. Scaling a body count would work on any shooter; driving a
 * million-bird mass into a ring, or keeping it out of one, only means anything
 * because the thing you are pushing flocks.
 */

export type ObjectiveKind = "disperse" | "drive" | "clear";

export interface Zone {
  x: number;
  y: number;
  r: number;
}

export interface Objective {
  kind: ObjectiveKind;
  /** Imperative, shown in the HUD. */
  label: string;
  target: number;
  zone: Zone | null;
  /** 0..1 */
  progress: number;
  resolved: boolean;
  success: boolean;
  /** Seconds spent in violation, for `clear`. */
  violation: number;
}

/** Seconds of violation tolerated before a `clear` objective is lost. */
const CLEAR_GRACE = 4;

const ORDER: ObjectiveKind[] = ["disperse", "drive", "clear"];

export function makeObjective(wave: number, world: [number, number]): Objective {
  const kind = ORDER[(wave - 1) % ORDER.length] as ObjectiveKind;
  const [wx, wy] = world;

  // Golden angle, so successive zones land far apart instead of clustering.
  const a = wave * 2.399963;
  const zone: Zone = {
    x: wx * (0.5 + Math.cos(a) * 0.3),
    y: wy * (0.5 + Math.sin(a) * 0.26),
    r: Math.min(wx, wy) * 0.17,
  };

  switch (kind) {
    case "disperse": {
      const target = 9000 + 4000 * (wave - 1);
      return base(kind, `Disperse ${target.toLocaleString("en-US")} birds`, target, null);
    }
    case "drive": {
      const target = 20000 + 7000 * (wave - 1);
      return base(kind, `Drive ${target.toLocaleString("en-US")} birds into the ring`, target, zone);
    }
    case "clear":
    default: {
      const target = Math.max(1200, 5000 - 500 * (wave - 1));
      return base(kind, `Hold the ring under ${target.toLocaleString("en-US")} birds`, target, zone);
    }
  }
}

function base(kind: ObjectiveKind, label: string, target: number, zone: Zone | null): Objective {
  return { kind, label, target, zone, progress: 0, resolved: false, success: false, violation: 0 };
}

export interface ObjectiveInput {
  dt: number;
  /** Birds dispersed since this wave began. */
  dispersedThisWave: number;
  /** Birds currently inside the zone, from the GPU census. */
  census: number;
}

export function updateObjective(o: Objective, input: ObjectiveInput): void {
  if (o.resolved) return;

  switch (o.kind) {
    case "disperse":
      o.progress = Math.min(1, input.dispersedThisWave / o.target);
      if (o.progress >= 1) {
        o.resolved = true;
        o.success = true;
      }
      break;

    case "drive":
      o.progress = Math.min(1, input.census / o.target);
      if (o.progress >= 1) {
        o.resolved = true;
        o.success = true;
      }
      break;

    case "clear":
      if (input.census > o.target) o.violation += input.dt;
      // Progress counts down the grace left, so the bar draining reads as
      // pressure rather than as something being accomplished.
      o.progress = Math.max(0, 1 - o.violation / CLEAR_GRACE);
      if (o.violation >= CLEAR_GRACE) {
        o.resolved = true;
        o.success = false;
      }
      break;
  }
}

/** Called when the wave timer runs out; unresolved objectives settle here. */
export function settleObjective(o: Objective): boolean {
  if (!o.resolved) {
    o.resolved = true;
    // `clear` is survived rather than completed: reaching the end without
    // burning the grace is a win. The other two had to be finished.
    o.success = o.kind === "clear";
  }
  return o.success;
}
