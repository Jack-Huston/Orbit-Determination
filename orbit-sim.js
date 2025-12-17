/* orbit-sim.js
   Orbit + EKF demo (2D, Earth-centric). No external libraries.
   - Physics: 2-body gravity + user thrust (acceleration) + optional process noise
   - Measurements: range, range-rate, bearing from visible ground stations + optional noise
   - Estimation: Extended Kalman Filter with linearized covariance propagation

   This file is written to be resilient: if a UI element is missing in the HTML,
   the simulator still runs and simply skips that feature.
*/
(() => {
  "use strict";

  // -----------------------------
  // DOM helpers
  // -----------------------------
  const $ = (sel, root = document) => root.querySelector(sel);

  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function fmt(x, digits = 3) {
    if (!Number.isFinite(x)) return "—";
    const ax = Math.abs(x);
    if (ax !== 0 && ax < 1e-3) return x.toExponential(2);
    if (ax > 1e6) return x.toExponential(2);
    return x.toFixed(digits);
  }
  function wrapToPi(a) {
    let x = a;
    while (x > Math.PI) x -= 2 * Math.PI;
    while (x < -Math.PI) x += 2 * Math.PI;
    return x;
  }

  // Deterministic RNG (Mulberry32) + Box-Muller normal
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a += 0x6D2B79F5;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeNormal(rng) {
    let spare = null;
    return function randn() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0, v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      const mag = Math.sqrt(-2.0 * Math.log(u));
      const z0 = mag * Math.cos(2 * Math.PI * v);
      const z1 = mag * Math.sin(2 * Math.PI * v);
      spare = z1;
      return z0;
    };
  }

  // -----------------------------
  // Small linear algebra (dense, Float64)
  // -----------------------------
  function zeros(r, c) {
    const A = new Array(r);
    for (let i = 0; i < r; i++) A[i] = new Float64Array(c);
    return A;
  }
  function eye(n) {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) A[i][i] = 1;
    return A;
  }
  function matCopy(A) {
    const r = A.length, c = A[0].length;
    const B = zeros(r, c);
    for (let i = 0; i < r; i++) B[i].set(A[i]);
    return B;
  }
  function matAdd(A, B) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) {
      const Ci = C[i], Ai = A[i], Bi = B[i];
      for (let j = 0; j < c; j++) Ci[j] = Ai[j] + Bi[j];
    }
    return C;
  }
  function matSub(A, B) {
    const r = A.length, c = A[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) {
      const Ci = C[i], Ai = A[i], Bi = B[i];
      for (let j = 0; j < c; j++) Ci[j] = Ai[j] - Bi[j];
    }
    return C;
  }
  function matMul(A, B) {
    const r = A.length, k = A[0].length, c = B[0].length;
    const C = zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
        C[i][j] = s;
      }
    }
    return C;
  }
  function matMulVec(A, x) {
    const r = A.length, c = A[0].length;
    const y = new Float64Array(r);
    for (let i = 0; i < r; i++) {
      let s = 0;
      for (let j = 0; j < c; j++) s += A[i][j] * x[j];
      y[i] = s;
    }
    return y;
  }
  function matTrans(A) {
    const r = A.length, c = A[0].length;
    const AT = zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) AT[j][i] = A[i][j];
    return AT;
  }
  function matScale(A, s) {
    const r = A.length, c = A[0].length;
    const B = zeros(r, c);
    for (let i = 0; i < r; i++) {
      const Bi = B[i], Ai = A[i];
      for (let j = 0; j < c; j++) Bi[j] = Ai[j] * s;
    }
    return B;
  }

  // Gaussian elimination with partial pivoting
  function matInv(A) {
    const n = A.length;
    const M = zeros(n, 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i][j] = A[i][j];
      M[i][n + i] = 1;
    }

    for (let col = 0; col < n; col++) {
      let pivotRow = col;
      let best = Math.abs(M[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(M[r][col]);
        if (v > best) { best = v; pivotRow = r; }
      }
      if (best < 1e-18) throw new Error("Singular matrix.");
      if (pivotRow !== col) {
        const tmp = M[col]; M[col] = M[pivotRow]; M[pivotRow] = tmp;
      }
      const piv = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= piv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }

    const Inv = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Inv[i][j] = M[i][n + j];
    return Inv;
  }

  function symmetrize(A) {
    const n = A.length;
    const B = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] = 0.5 * (A[i][j] + A[j][i]);
    return B;
  }

  // -----------------------------
  // Stats: normal quantile & chi-square bounds (Wilson-Hilferty)
  // -----------------------------
  function normInv(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const plow = 0.02425;
    const phigh = 1 - plow;

    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;

    let q, r;
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    }
    if (p > phigh) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
    }
    q = p - 0.5;
    r = q*q;
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
  }
  function chi2Inv(p, k) {
    const z = normInv(p);
    const a = 2 / (9 * k);
    const q = k * Math.pow(1 - a + z * Math.sqrt(a), 3);
    return Math.max(0, q);
  }

  // -----------------------------
  // Earth constants (not user-tunable)
  // -----------------------------
  const CONST = Object.freeze({
    mu: 398600.0,          // km^3/s^2
    Re: 6378.0,            // km
    omega: 2 * Math.PI / 86400, // rad/s
    dt: 10.0,              // seconds per step (measurement cadence)
    minAlt: 120.0          // km safety margin for periapsis
  });

  // -----------------------------
  // Dynamics + measurement models
  // -----------------------------
  function orbitDeriv(x, u, mu) {
    // x = [X, Xdot, Y, Ydot]
    const X = x[0], Xd = x[1], Y = x[2], Yd = x[3];
    const r2 = X*X + Y*Y;
    const r = Math.sqrt(r2);
    const r3 = r2 * r + 1e-18;
    const ax = -mu * X / r3 + u[0];
    const ay = -mu * Y / r3 + u[1];
    return new Float64Array([Xd, ax, Yd, ay]);
  }

  function rk4Step(x, u, dt, mu) {
    const k1 = orbitDeriv(x, u, mu);
    const x2 = new Float64Array(4);
    for (let i = 0; i < 4; i++) x2[i] = x[i] + 0.5 * dt * k1[i];

    const k2 = orbitDeriv(x2, u, mu);
    const x3 = new Float64Array(4);
    for (let i = 0; i < 4; i++) x3[i] = x[i] + 0.5 * dt * k2[i];

    const k3 = orbitDeriv(x3, u, mu);
    const x4 = new Float64Array(4);
    for (let i = 0; i < 4; i++) x4[i] = x[i] + dt * k3[i];

    const k4 = orbitDeriv(x4, u, mu);
    const out = new Float64Array(4);
    for (let i = 0; i < 4; i++) out[i] = x[i] + (dt / 6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    return out;
  }

  function A_jacobian(x, mu) {
    const x1 = x[0], x3 = x[2];
    const r = Math.sqrt(x1*x1 + x3*x3) + 1e-18;
    const r5 = Math.pow(r, 5);

    const A_11 = mu * (2*x1*x1 - x3*x3) / r5;
    const A_13 = 3 * mu * x1 * x3 / r5;
    const A_31 = A_13;
    const A_33 = mu * (2*x3*x3 - x1*x1) / r5;

    const A = zeros(4,4);
    A[0][1] = 1;
    A[1][0] = A_11; A[1][2] = A_13;
    A[2][3] = 1;
    A[3][0] = A_31; A[3][2] = A_33;
    return A;
  }

  function stationState(i, t, Re, omega) {
    // 12 stations equally spaced in longitude
    const theta0 = i * Math.PI / 6;
    const th = omega * t + theta0;
    const c = Math.cos(th), s = Math.sin(th);
    const X = Re * c;
    const Y = Re * s;
    const Xd = -omega * Re * s;
    const Yd = omega * Re * c;
    return { X, Y, Xd, Yd, th };
  }

  function visibleStations(x, t, Re, omega) {
    // Horizon test: (r_sc - r_st) · r_st > 0
    const vis = [];
    const X = x[0], Y = x[2];
    for (let i = 0; i < 12; i++) {
      const s = stationState(i, t, Re, omega);
      const dx = X - s.X;
      const dy = Y - s.Y;
      const dot = dx * s.X + dy * s.Y;
      if (dot > 0) vis.push(i+1); // 1..12
    }
    return vis;
  }

  function measureOneStation(x, t, stationId, Re, omega) {
    const i = stationId - 1;
    const s = stationState(i, t, Re, omega);

    const x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    const dX = x1 - s.X;
    const dY = x3 - s.Y;
    const dXd = x2 - s.Xd;
    const dYd = x4 - s.Yd;

    const rho = Math.sqrt(dX*dX + dY*dY) + 1e-18;
    const rhod = (dX*dXd + dY*dYd) / rho;
    const phi = Math.atan2(dY, dX);

    return { rho, rhod, phi, dX, dY, dXd, dYd };
  }

  function H_jacobian_forStations(x, t, stationIds, Re, omega) {
    const n = stationIds.length;
    const yhat = new Float64Array(3*n);
    const H = zeros(3*n, 4);

    for (let k = 0; k < n; k++) {
      const id = stationIds[k];
      const m = measureOneStation(x, t, id, Re, omega);

      const rho = m.rho;
      const a = m.dX*m.dXd + m.dY*m.dYd;
      const rho3 = rho*rho*rho;

      yhat[3*k + 0] = m.rho;
      yhat[3*k + 1] = m.rhod;
      yhat[3*k + 2] = m.phi;

      // Range
      H[3*k + 0][0] = m.dX / rho;
      H[3*k + 0][2] = m.dY / rho;

      // Range-rate
      H[3*k + 1][0] = (m.dXd / rho) - (a * m.dX / rho3);
      H[3*k + 1][2] = (m.dYd / rho) - (a * m.dY / rho3);
      H[3*k + 1][1] = m.dX / rho;
      H[3*k + 1][3] = m.dY / rho;

      // Bearing
      const rho2 = rho*rho;
      H[3*k + 2][0] = -m.dY / rho2;
      H[3*k + 2][2] =  m.dX / rho2;
    }

    return { yhat, H };
  }

  // Process noise mapping: accel noise -> state
  // L = [[dt^2/2,0],[dt,0],[0,dt^2/2],[0,dt]]
  function QdFromQacc(Qacc2x2, dt) {
    const h = 0.5 * dt * dt;
    const L = zeros(4,2);
    L[0][0] = h;   L[1][0] = dt;
    L[2][1] = h;   L[3][1] = dt;

    const LQ = matMul(L, Qacc2x2);
    const Qd = matMul(LQ, matTrans(L));
    return symmetrize(Qd);
  }

  // -----------------------------
  // EKF
  // -----------------------------
  class EKF {
    constructor(mu, Re, omega) {
      this.mu = mu;
      this.Re = Re;
      this.omega = omega;

      this.x = new Float64Array(4);
      this.P = eye(4);

      this.Qacc = eye(2);
      this.Rstn = eye(3);

      this.lastInnov = null;
      this.lastS = null;
      this.lastYhat = null;
      this.lastStationIds = [];
      this.lastNEES = NaN;
      this.lastNIS = NaN;
    }

    setQR(Qacc2, Rstn3) {
      this.Qacc = Qacc2;
      this.Rstn = Rstn3;
    }

    reset(x0, P0) {
      this.x = new Float64Array(x0);
      this.P = matCopy(P0);
      this.lastInnov = null;
      this.lastS = null;
      this.lastYhat = null;
      this.lastStationIds = [];
      this.lastNEES = NaN;
      this.lastNIS = NaN;
    }

    step(dt, tNext, u, yMeas, stationIds) {
      const xminus = rk4Step(this.x, u, dt, this.mu);

      const A = A_jacobian(this.x, this.mu);
      const F = matAdd(eye(4), matScale(A, dt));
      const Qd = QdFromQacc(this.Qacc, dt);

      let Pminus = matAdd(matMul(matMul(F, this.P), matTrans(F)), Qd);
      Pminus = symmetrize(Pminus);

      if (!yMeas || yMeas.length === 0 || stationIds.length === 0) {
        this.x = xminus;
        this.P = Pminus;
        this.lastInnov = null;
        this.lastS = null;
        this.lastYhat = null;
        this.lastStationIds = [];
        return;
      }

      const { yhat, H } = H_jacobian_forStations(xminus, tNext, stationIds, this.Re, this.omega);

      const n = stationIds.length;
      const m = 3*n;
      const R = zeros(m, m);
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          R[3*k + i][3*k + j] = this.Rstn[i][j];
        }
      }

      const e = new Float64Array(m);
      for (let i = 0; i < m; i++) e[i] = yMeas[i] - yhat[i];
      for (let k = 0; k < n; k++) e[3*k + 2] = wrapToPi(e[3*k + 2]);

      const HP = matMul(H, Pminus);
      const S = symmetrize(matAdd(matMul(HP, matTrans(H)), R));

      const PHt = matMul(Pminus, matTrans(H));
      const Sinv = matInv(S);
      const K = matMul(PHt, Sinv);

      const Ke = matMulVec(K, e);
      const xplus = new Float64Array(4);
      for (let i = 0; i < 4; i++) xplus[i] = xminus[i] + Ke[i];

      const KH = matMul(K, H);
      const IKH = matSub(eye(4), KH);
      let Pplus = matMul(IKH, Pminus);
      Pplus = symmetrize(Pplus);

      this.x = xplus;
      this.P = Pplus;

      this.lastInnov = e;
      this.lastS = S;
      this.lastYhat = yhat;
      this.lastStationIds = stationIds.slice();
    }

    computeNEES(trueX) {
      const e = new Float64Array(4);
      for (let i = 0; i < 4; i++) e[i] = this.x[i] - trueX[i];
      const Pinv = matInv(this.P);

      let s = 0;
      for (let i = 0; i < 4; i++) {
        let row = 0;
        for (let j = 0; j < 4; j++) row += Pinv[i][j] * e[j];
        s += e[i] * row;
      }
      this.lastNEES = s;
      return s;
    }

    computeNIS() {
      if (!this.lastInnov || !this.lastS) {
        this.lastNIS = NaN;
        return NaN;
      }
      const Sinv = matInv(this.lastS);
      const v = this.lastInnov;
      const m = v.length;
      let s = 0;
      for (let i = 0; i < m; i++) {
        let row = 0;
        for (let j = 0; j < m; j++) row += Sinv[i][j] * v[j];
        s += v[i] * row;
      }
      this.lastNIS = s;
      return s;
    }
  }

  // -----------------------------
  // Canvas utilities
  // -----------------------------
  function setupCanvasHiDPI(canvas) {
    if (!canvas) return null;
    const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(2, Math.floor(rect.width * dpr));
    const h = Math.max(2, Math.floor(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function clearPanel(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const g = ctx.createRadialGradient(w*0.5, h*0.4, 10, w*0.5, h*0.5, Math.max(w,h)*0.8);
    g.addColorStop(0, "rgba(58,166,255,0.06)");
    g.addColorStop(1, "rgba(0,0,0,0.0)");
    ctx.fillStyle = g;
    ctx.fillRect(0,0,w,h);
  }

  // Axes w/ ticks (linear)
  function drawAxesWithTicks(ctx, x, y, w, h, opts) {
    const {
      title = "",
      xLabel = "",
      yLabel = "",
      xMin = 0,
      xMax = 1,
      yMin = 0,
      yMax = 1,
      xTicks = 5,
      yTicks = 4,
    } = opts || {};

    ctx.save();
    ctx.translate(x, y);

    // frame
    ctx.strokeStyle = "rgba(110,231,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);

    // grid + ticks
    ctx.font = `600 11px ${getComputedStyle(document.documentElement).getPropertyValue("--mono") || "ui-monospace, Menlo, Consolas, monospace"}`;
    ctx.fillStyle = "rgba(127,138,161,0.92)";
    ctx.strokeStyle = "rgba(110,231,255,0.06)";
    ctx.lineWidth = 1;

    for (let i = 0; i <= xTicks; i++) {
      const u = i / xTicks;
      const gx = u * w;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
      ctx.stroke();

      const val = xMin + u * (xMax - xMin);
      const s = fmt(val, Math.abs(xMax - xMin) < 10 ? 2 : 0);
      ctx.fillText(s, gx + 2, h - 6);
    }
    for (let j = 0; j <= yTicks; j++) {
      const u = j / yTicks;
      const gy = h - u * h;
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();

      const val = yMin + u * (yMax - yMin);
      const s = fmt(val, Math.abs(yMax - yMin) < 10 ? 2 : 0);
      ctx.fillText(s, 6, gy - 2);
    }

    // titles/labels
    ctx.fillStyle = "rgba(223,244,255,0.9)";
    ctx.font = "800 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillText(title, 10, 16);

    ctx.fillStyle = "rgba(127,138,161,0.92)";
    ctx.font = "700 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    if (xLabel) ctx.fillText(xLabel, w - ctx.measureText(xLabel).width - 10, h - 8);
    if (yLabel) {
      ctx.save();
      ctx.translate(10, h/2);
      ctx.rotate(-Math.PI/2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    ctx.restore();
  }

  function computeMinMax(seriesList) {
    let mn = Infinity, mx = -Infinity;
    for (const s of seriesList) {
      if (!s) continue;
      for (let i = 0; i < s.length; i++) {
        const v = s[i];
        if (!Number.isFinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) return { mn: -1, mx: 1 };
    if (mn === mx) return { mn: mn - 1, mx: mx + 1 };
    const pad = 0.08 * (mx - mn);
    return { mn: mn - pad, mx: mx + pad };
  }

  function plotLine(ctx, x0, y0, w, h, t, y, tMin, tMax, yMin, yMax, style) {
    if (!t || !y || t.length < 2) return;
    const n = Math.min(t.length, y.length);
    const denomT = (tMax - tMin) || 1;
    const denomY = (yMax - yMin) || 1;
    const sx = (tt) => x0 + (tt - tMin) / denomT * w;
    const sy = (vv) => y0 + h - (vv - yMin) / denomY * h;

    ctx.save();
    ctx.strokeStyle = style.stroke || "rgba(231,238,252,0.95)";
    ctx.lineWidth = style.width || 1.5;
    if (style.dash) ctx.setLineDash(style.dash);
    ctx.beginPath();
    ctx.moveTo(sx(t[0]), sy(y[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(sx(t[i]), sy(y[i]));
    ctx.stroke();
    ctx.restore();
  }

  function plotScatter(ctx, x0, y0, w, h, t, y, tMin, tMax, yMin, yMax, style) {
    if (!t || !y || t.length < 1) return;
    const n = Math.min(t.length, y.length);
    const denomT = (tMax - tMin) || 1;
    const denomY = (yMax - yMin) || 1;
    const sx = (tt) => x0 + (tt - tMin) / denomT * w;
    const sy = (vv) => y0 + h - (vv - yMin) / denomY * h;

    const r = style.r || 2.2;
    ctx.save();
    ctx.fillStyle = style.fill || "rgba(58,166,255,0.95)";
    for (let i = 0; i < n; i++) {
      const tt = t[i];
      if (tt < tMin || tt > tMax) continue;
      const px = sx(tt);
      const py = sy(y[i]);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2*Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // -----------------------------
  // Orbit view camera
  // -----------------------------
  class Camera {
    constructor() {
      this.cx = 0;
      this.cy = 0;
      this.kmPerPx = 35;
    }
    reset(Re) {
      this.cx = 0;
      this.cy = 0;
      this.kmPerPx = Math.max(10, Re / 140);
    }
    worldToScreen(wx, wy, w, h) {
      const px = (wx - this.cx) / this.kmPerPx + w/2;
      const py = (wy - this.cy) / this.kmPerPx + h/2;
      return { x: px, y: py };
    }
    screenToWorld(px, py, w, h) {
      const wx = (px - w/2) * this.kmPerPx + this.cx;
      const wy = (py - h/2) * this.kmPerPx + this.cy;
      return { x: wx, y: wy };
    }
  }

  // -----------------------------
  // Plot view (time window pan/zoom when paused)
  // -----------------------------
  class TimeView {
    constructor() {
      this.windowSec = 1200; // default visible window
      this.center = 0;       // center time (sec), used when paused
      this.locked = false;   // becomes true after user interacts while paused
    }
    follow(tNow) {
      if (this.locked) return;
      this.center = tNow;
    }
    resetFollow() {
      this.locked = false;
    }
    getRange(tNow) {
      const c = this.locked ? this.center : tNow;
      const half = 0.5 * this.windowSec;
      return { tMin: Math.max(0, c - half), tMax: Math.max(1, c + half) };
    }
    zoom(factor) {
      const w = this.windowSec;
      this.windowSec = clamp(w / factor, 120, 7200);
    }
    pan(dt) {
      this.center = Math.max(0, this.center + dt);
    }
    userTouched() { this.locked = true; }
  }

  // -----------------------------
  // UI handles (nullable)
  // -----------------------------
  const el = {
    statusRun: $("#statusRun"),
    readoutTime: $("#readoutTime"),
    readoutStep: $("#readoutStep"),
    readoutVisible: $("#readoutVisible"),

    roRho: $("#roRho"),
    roRhod: $("#roRhod"),
    roPhi: $("#roPhi"),
    roNEES: $("#roNEES"),
    roNIS: $("#roNIS"),
    roErrNorm: $("#roErrNorm"),

    stationBoard: $("#stationBoard"),

    btnToggleRun: $("#btnToggleRun"),
    btnStep: $("#btnStep"),
    btnReset: $("#btnReset"),
    btnApplyIC: $("#btnApplyIC"),
    btnResetFilter: $("#btnResetFilter"),

    // Inputs (many may be absent after HTML simplification)
    inpSteps: $("#inpSteps"),

    // Orbital elements (expected)
    inpRpAlt: $("#inpRpAlt"),       // km
    inpEcc: $("#inpEcc"),           // unitless [0,1)
    inpNuDeg: $("#inpNuDeg"),       // deg
    inpArgPerDeg: $("#inpArgPerDeg"), // deg (optional)
    inpM0Deg: $("#inpM0Deg"),       // deg (optional, if you choose mean anomaly)
    inpThrustMag: $("#inpThrustMag"), // km/s^2 (or m/s^2 converted)
    inpThrustMagMS2: $("#inpThrustMagMS2"), // m/s^2 (optional)

    // Noise toggles & scalars
    tglProcNoise: $("#tglProcNoise"),
    tglMeasNoise: $("#tglMeasNoise"),

    inpQtrueAx: $("#inpQtrueAx"),
    inpQtrueAy: $("#inpQtrueAy"),
    inpRtrueRho: $("#inpRtrueRho"),
    inpRtrueRhod: $("#inpRtrueRhod"),
    inpRtruePhi: $("#inpRtruePhi"),

    inpSigX0: $("#inpSigX0"),
    inpSigV0: $("#inpSigV0"),
    inpQkfAx: $("#inpQkfAx"),
    inpQkfAy: $("#inpQkfAy"),
    inpRkfRho: $("#inpRkfRho"),
    inpRkfRhod: $("#inpRkfRhod"),
    inpRkfPhi: $("#inpRkfPhi"),
    inpAlpha: $("#inpAlpha"),

    // Thrust buttons (expected)
    thrUp: $("#thrUp"),
    thrDown: $("#thrDown"),
    thrLeft: $("#thrLeft"),
    thrRight: $("#thrRight"),
    thrZero: $("#thrZero"),

    // Orbit view
    orbitCanvas: $("#orbitCanvas"),
    btnCenterView: $("#btnCenterView"),
    btnResetView: $("#btnResetView"),
    btnZoomToSat: $("#btnZoomToSat"),

    // Plots
    plotStates: $("#plotStates"),
    plotErrors: $("#plotErrors"),
    plotMeas: $("#plotMeas"),
    plotCons: $("#plotCons"),
  };

  // -----------------------------
  // Station colors (should match CSS intent)
  // -----------------------------
  const STATION_COLORS = [
    "#6ee7ff", "#78f3b6", "#ffcc66", "#ff5c7a",
    "#a78bfa", "#22d3ee", "#fb7185", "#34d399",
    "#fbbf24", "#60a5fa", "#f472b6", "#a3e635"
  ];
  function stationColor(id) {
    const i = (id - 1) % STATION_COLORS.length;
    return STATION_COLORS[i];
  }

  // -----------------------------
  // App state
  // -----------------------------
  const app = {
    running: false,
    speed: 10,
    dt: CONST.dt,
    Kmax: 1_000_000,

    mu: CONST.mu,
    Re: CONST.Re,
    omega: CONST.omega,

    // user command acceleration (km/s^2)
    uCmd: new Float64Array([0, 0]),
    uHold: { up:false, down:false, left:false, right:false },
    thrustMag: 0.0, // km/s^2

    // time/step
    t: 0,
    k: 0,

    truth: new Float64Array(4),
    ekf: new EKF(CONST.mu, CONST.Re, CONST.omega),

    enableProcNoise: true,
    enableMeasNoise: true,

    // Defaults intentionally “noisier” to match typical classroom/testing demos
    QtrueAcc: (() => { const Q = zeros(2,2); Q[0][0]=1e-8; Q[1][1]=1e-8; return Q; })(),
    RtrueStn: (() => { const R = zeros(3,3); R[0][0]=5*5; R[1][1]=0.01*0.01; R[2][2]=0.01*0.01; return R; })(),

    bufMax: 2400,
    tHist: [],
    xTrueHist: [[],[],[],[]],
    xHatHist:  [[],[],[],[]],
    sigHist:   [[],[],[],[]],
    errHist:   [[],[],[],[]],
    neesHist: [],
    nisHist: [],
    nisDfHist: [],

    // Measurements organized by station for coloring
    measByStation: Array.from({length:12}, () => ({ t:[], rho:[], rhod:[], phi:[] })),
    predByStation: Array.from({length:12}, () => ({ t:[], rho:[], rhod:[], phi:[] })),

    pulses: [], // {t0,dur,sx,sy,tx,ty,id,color}

    cam: new Camera(),
    orbitCtx: null,

    ctxStates: null,
    ctxErrors: null,
    ctxMeas: null,
    ctxCons: null,

    plotDirty: true,
    lastPlotDraw: 0,

    // view controllers
    viewMeas: new TimeView(),
    viewCons: new TimeView(),
    viewStates: new TimeView(),
    viewErrors: new TimeView(),

    // RNG
    rng: null,
    randn: null,

    // Safety latch
    lastSafeState: null, // {t,k,truth,ekfX,ekfP}
  };

  // -----------------------------
  // Station board UI
  // -----------------------------
  function makeDishSVG() {
    // Dish facing up (no upside-down look). Kept minimal and crisp.
    return `
      <svg viewBox="0 0 120 34" aria-hidden="true" focusable="false">
        <path class="dishfill" d="M 12 26 Q 60 6 108 26 Q 60 30 12 26 Z"></path>
        <path class="dishline" d="M 12 26 Q 60 6 108 26"></path>
        <path class="dishline" d="M 58 26 L 52 10"></path>
        <path class="dishline" d="M 60 26 L 60 9"></path>
        <path class="dishline" d="M 62 26 L 68 10"></path>
        <path class="dishline" d="M 54 26 L 54 32"></path>
        <path class="dishline" d="M 66 26 L 66 32"></path>
      </svg>
    `;
  }
  function makeLightningSVG() {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M13 2L3 14h7l-1 8 12-15h-7l-1-5z"></path>
      </svg>
    `;
  }
  function makeSquiggleSVG() {
    return `
      <svg viewBox="0 0 120 18" aria-hidden="true" focusable="false">
        <path class="squigglepath"
              d="M2 10 C 10 2, 18 16, 26 10
                 S 42 2, 50 10
                 S 66 16, 74 10
                 S 90 2, 98 10
                 S 110 16, 118 10"></path>
      </svg>
    `;
  }

  function buildStationBoard() {
    if (!el.stationBoard) return;
    el.stationBoard.innerHTML = "";
    for (let i = 1; i <= 12; i++) {
      const tile = document.createElement("div");
      tile.className = "stationtile";
      tile.id = `st-${i}`;
      tile.dataset.station = String(i);
      tile.style.setProperty("--stc", stationColor(i));
      tile.innerHTML = `
        <div class="stationhead">
          <div class="stationname">STATION ${String(i).padStart(2,"0")}</div>
          <div class="stationmeta" id="stmeta-${i}">—</div>
        </div>
        <div class="dishicon">${makeDishSVG()}</div>
        <div class="datasquiggle">${makeSquiggleSVG()}</div>
        <div class="lightning">${makeLightningSVG()}</div>
      `;
      el.stationBoard.appendChild(tile);
    }
  }

  function updateStationBoard(visibleIds, pulsingIds) {
    if (!el.stationBoard) return;
    const visSet = new Set(visibleIds);
    const pulseSet = new Set(pulsingIds);

    for (let i = 1; i <= 12; i++) {
      const tile = $(`#st-${i}`);
      if (!tile) continue;
      tile.classList.toggle("is-visible", visSet.has(i));
      tile.classList.toggle("is-pulsing", pulseSet.has(i));
      const meta = $(`#stmeta-${i}`);
      if (meta) meta.textContent = visSet.has(i) ? `VISIBLE` : `—`;
    }
  }

  // -----------------------------
  // Matrix builders from UI (with sensible fallbacks)
  // -----------------------------
  function buildP0(sigX, sigV) {
    const P0 = zeros(4,4);
    P0[0][0] = sigX*sigX;
    P0[1][1] = sigV*sigV;
    P0[2][2] = sigX*sigX;
    P0[3][3] = sigV*sigV;
    return P0;
  }
  function buildQacc(ax, ay) {
    const Q = zeros(2,2);
    Q[0][0] = ax;
    Q[1][1] = ay;
    return Q;
  }
  function buildRstn(sigRho, sigRhod, sigPhi) {
    const R = zeros(3,3);
    R[0][0] = sigRho*sigRho;
    R[1][1] = sigRhod*sigRhod;
    R[2][2] = sigPhi*sigPhi;
    return R;
  }

  // -----------------------------
  // Orbital elements -> Cartesian (2D)
  // Inputs:
  //   rpAlt (km), e [0,1), nuDeg, argPerDeg (optional)
  // Safety: clamp periapsis to > Re + minAlt.
  // -----------------------------
  function elementsToState2D(rpAltKm, e, nuDeg, argPerDeg) {
    const rp = CONST.Re + Math.max(CONST.minAlt, rpAltKm);
    const ecc = clamp(e, 0, 0.999);
    const nu = (nuDeg * Math.PI) / 180;
    const arg = (argPerDeg || 0) * Math.PI / 180;

    // a = rp/(1-e)
    const a = rp / (1 - ecc);
    const p = a * (1 - ecc*ecc);

    // position/velocity in perifocal frame
    const r = p / (1 + ecc * Math.cos(nu));
    const xP = r * Math.cos(nu);
    const yP = r * Math.sin(nu);

    const vScale = Math.sqrt(CONST.mu / p);
    const vxP = -vScale * Math.sin(nu);
    const vyP =  vScale * (ecc + Math.cos(nu));

    // rotate by argument of periapsis
    const c = Math.cos(arg), s = Math.sin(arg);
    const x = c*xP - s*yP;
    const y = s*xP + c*yP;
    const vx = c*vxP - s*vyP;
    const vy = s*vxP + c*vyP;

    return new Float64Array([x, vx, y, vy]);
  }

  function statePeriapsisRadius(x) {
    // For a general state, estimate rp from energy & angular momentum (ellipse assumed)
    const X=x[0], Y=x[2], Vx=x[1], Vy=x[3];
    const r = Math.hypot(X,Y);
    const v2 = Vx*Vx + Vy*Vy;
    const h = X*Vy - Y*Vx; // scalar
    const eps = v2/2 - CONST.mu / (r + 1e-18);
    if (eps >= 0) return NaN; // not bound
    const a = -CONST.mu / (2*eps);
    const e = Math.sqrt(Math.max(0, 1 - (h*h)/(CONST.mu*a)));
    const rp = a*(1 - e);
    return rp;
  }

  // -----------------------------
  // Running / speed
  // -----------------------------
  function setRunning(run) {
    app.running = run;
    if (el.statusRun) {
      el.statusRun.textContent = run ? "RUNNING" : "PAUSED";
      el.statusRun.style.borderColor = run ? "rgba(120,243,182,.35)" : "rgba(110,231,255,.22)";
    }
    if (el.btnToggleRun) el.btnToggleRun.textContent = run ? "Pause" : "Run";

    // When resuming, return plots to follow mode
    if (run) {
      app.viewMeas.resetFollow();
      app.viewCons.resetFollow();
      app.viewStates.resetFollow();
      app.viewErrors.resetFollow();
    }
  }

  function setSpeed(s) {
    app.speed = s;
    for (const id of ["spd1","spd10","spd100","spd1000"]) {
      const b = $(`#${id}`);
      if (!b) continue;
      b.classList.toggle("is-active", parseFloat(b.dataset.speed) === s);
    }
  }

  // -----------------------------
  // History buffers
  // -----------------------------
  function pushLimited(arr, val, maxN = app.bufMax) {
    arr.push(val);
    if (arr.length > maxN) arr.shift();
  }
  function pushHistorySample() {
    const t = app.t;
    const xt = app.truth;
    const xh = app.ekf.x;

    pushLimited(app.tHist, t);
    for (let i = 0; i < 4; i++) {
      pushLimited(app.xTrueHist[i], xt[i]);
      pushLimited(app.xHatHist[i], xh[i]);
      const s = Math.sqrt(Math.max(0, app.ekf.P[i][i]));
      pushLimited(app.sigHist[i], s);
      pushLimited(app.errHist[i], xh[i] - xt[i]);
    }
    const nees = app.ekf.computeNEES(xt);
    const nis = app.ekf.computeNIS();
    pushLimited(app.neesHist, nees);
    pushLimited(app.nisHist, nis);
    pushLimited(app.nisDfHist, app.ekf.lastInnov ? app.ekf.lastInnov.length : NaN);

    app.plotDirty = true;
  }

  // -----------------------------
  // MVN sampling (small symmetric PSD)
  // -----------------------------
  function sampleMVN(dim, cov, randn) {
    const n = dim;
    const A = matCopy(cov);
    for (let i = 0; i < n; i++) A[i][i] += 1e-18;

    const L = zeros(n,n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k]*L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(0, s));
        else L[i][j] = s / (L[j][j] + 1e-18);
      }
    }

    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) z[i] = randn();
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j <= i; j++) s += L[i][j]*z[j];
      x[i] = s;
    }
    return x;
  }

  // -----------------------------
  // Readouts
  // -----------------------------
  function updateReadouts(visibleIds) {
    if (el.readoutTime) el.readoutTime.textContent = `${fmt(app.t, 1)} s`;
    if (el.readoutStep) el.readoutStep.textContent = `${app.k}`;
    if (el.readoutVisible) el.readoutVisible.textContent = `${visibleIds.length}`;

    // Display the most recent measurement (any station)
    let latest = null;
    for (let sid = 1; sid <= 12; sid++) {
      const m = app.measByStation[sid-1];
      if (m.t.length === 0) continue;
      const i = m.t.length - 1;
      if (!latest || m.t[i] > latest.t) {
        latest = { t: m.t[i], rho: m.rho[i], rhod: m.rhod[i], phi: m.phi[i] };
      }
    }

    if (latest) {
      if (el.roRho) el.roRho.textContent = `${fmt(latest.rho, 3)} km`;
      if (el.roRhod) el.roRhod.textContent = `${fmt(latest.rhod, 5)} km/s`;
      if (el.roPhi) el.roPhi.textContent = `${fmt(latest.phi, 5)} rad`;
    } else {
      if (el.roRho) el.roRho.textContent = "—";
      if (el.roRhod) el.roRhod.textContent = "—";
      if (el.roPhi) el.roPhi.textContent = "—";
    }

    if (el.roNEES) el.roNEES.textContent = Number.isFinite(app.ekf.lastNEES) ? fmt(app.ekf.lastNEES, 2) : "—";
    if (el.roNIS) el.roNIS.textContent = Number.isFinite(app.ekf.lastNIS) ? fmt(app.ekf.lastNIS, 2) : "—";

    const dx = app.ekf.x[0] - app.truth[0];
    const dy = app.ekf.x[2] - app.truth[2];
    const dvx = app.ekf.x[1] - app.truth[1];
    const dvy = app.ekf.x[3] - app.truth[3];
    const nrm = Math.sqrt(dx*dx + dy*dy + dvx*dvx + dvy*dvy);
    if (el.roErrNorm) el.roErrNorm.textContent = fmt(nrm, 4);
  }

  // -----------------------------
  // Orbit view rendering
  // -----------------------------
  function drawCovEllipse(ctx, P, xhat, W, H) {
    const Pxx = P[0][0], Pxy = P[0][2], Pyy = P[2][2];
    if (!Number.isFinite(Pxx) || !Number.isFinite(Pyy)) return;

    const tr = Pxx + Pyy;
    const det = Pxx*Pyy - Pxy*Pxy;
    const disc = Math.max(0, tr*tr/4 - det);
    const l1 = tr/2 + Math.sqrt(disc);
    const l2 = tr/2 - Math.sqrt(disc);

    const angle = 0.5 * Math.atan2(2*Pxy, (Pxx - Pyy));
    const s1 = Math.sqrt(Math.max(0, l1));
    const s2 = Math.sqrt(Math.max(0, l2));
    const k = 2;
    const rx = (k*s1) / app.cam.kmPerPx;
    const ry = (k*s2) / app.cam.kmPerPx;

    const c = app.cam.worldToScreen(xhat[0], xhat[2], W, H);

    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(angle);
    ctx.strokeStyle = "rgba(110,231,255,0.30)";
    ctx.fillStyle = "rgba(110,231,255,0.08)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(0, rx), Math.max(0, ry), 0, 0, 2*Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawEstimatedOneOrbitPreview(ctx, W, H) {
    // Propagate EKF state forward with zero thrust (preview only)
    const x0 = app.ekf.x;
    const X=x0[0], Y=x0[2], Vx=x0[1], Vy=x0[3];
    const r = Math.hypot(X,Y);
    const v2 = Vx*Vx + Vy*Vy;
    const eps = v2/2 - CONST.mu / (r + 1e-18);
    if (!(eps < 0)) return;

    const a = -CONST.mu/(2*eps);
    const T = 2*Math.PI*Math.sqrt((a*a*a)/CONST.mu);
    const horizon = clamp(T, 900, 20000);

    const nPts = 220;
    const dt = horizon / nPts;

    let x = new Float64Array(x0);
    ctx.save();
    ctx.strokeStyle = "rgba(110,231,255,0.35)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();

    for (let i = 0; i <= nPts; i++) {
      const p = app.cam.worldToScreen(x[0], x[2], W, H);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
      x = rk4Step(x, new Float64Array([0,0]), dt, CONST.mu);
    }

    ctx.stroke();
    ctx.restore();
  }

  function drawOrbitView() {
    const canvas = el.orbitCanvas;
    const ctx = app.orbitCtx;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0,0,W,H);

    // Earth
    const earth = app.cam.worldToScreen(0,0,W,H);
    const RePx = app.Re / app.cam.kmPerPx;
    ctx.save();
    ctx.beginPath();
    ctx.arc(earth.x, earth.y, Math.max(0, RePx), 0, 2*Math.PI);
    ctx.fillStyle = "rgba(120,243,182,0.10)";
    ctx.fill();
    ctx.strokeStyle = "rgba(120,243,182,0.22)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();

    // Stations
    for (let i = 1; i <= 12; i++) {
      const s = stationState(i - 1, app.t, app.Re, app.omega);
      const p = app.cam.worldToScreen(s.X, s.Y, W, H);
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.2, 0, 2*Math.PI);
      ctx.fillStyle = stationColor(i);
      ctx.globalAlpha = 0.90;
      ctx.fill();
      ctx.restore();
    }

    // Trajectories (truth + estimate)
    const n = app.tHist.length;
    if (n > 2) {
      ctx.save();
      ctx.strokeStyle = "rgba(231,238,252,0.90)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const X = app.xTrueHist[0][i];
        const Y = app.xTrueHist[2][i];
        const p = app.cam.worldToScreen(X, Y, W, H);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "rgba(110,231,255,0.90)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const X = app.xHatHist[0][i];
        const Y = app.xHatHist[2][i];
        if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
        const p = app.cam.worldToScreen(X, Y, W, H);
        if (!started) { ctx.moveTo(p.x, p.y); started = true; }
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 1-orbit preview from EKF state (dashed)
    drawEstimatedOneOrbitPreview(ctx, W, H);

    // Current markers
    const satT = app.cam.worldToScreen(app.truth[0], app.truth[2], W, H);
    const satE = app.cam.worldToScreen(app.ekf.x[0], app.ekf.x[2], W, H);

    ctx.save();
    ctx.beginPath();
    ctx.arc(satT.x, satT.y, 4.8, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(231,238,252,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(231,238,252,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(satE.x, satE.y, 4.8, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(110,231,255,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(110,231,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Cov ellipse
    drawCovEllipse(ctx, app.ekf.P, app.ekf.x, W, H);

    // Signal pulses (measurement link visualization, colored per station)
    const now = performance.now();
    const still = [];
    for (const p of app.pulses) {
      const age = now - p.t0;
      if (age > p.dur) continue;
      still.push(p);

      const a = age / p.dur;
      const s0 = app.cam.worldToScreen(p.sx, p.sy, W, H);
      const s1 = app.cam.worldToScreen(p.tx, p.ty, W, H);
      const x = s0.x + a * (s1.x - s0.x);
      const y = s0.y + a * (s1.y - s0.y);

      ctx.save();
      ctx.strokeStyle = hexToRgba(p.color, 0.22 * (1-a) + 0.06);
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.stroke();

      // squiggle-ish stroke overlay
      ctx.strokeStyle = hexToRgba(p.color, 0.20 * (1-a) + 0.06);
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = hexToRgba(p.color, 0.85 * (1-a) + 0.15);
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, 2*Math.PI);
      ctx.fill();
      ctx.restore();
    }
    app.pulses = still;

    // HUD
    ctx.save();
    ctx.fillStyle = "rgba(223,244,255,0.92)";
    ctx.font = "800 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`t = ${fmt(app.t,1)} s`, 10, 18);
    ctx.fillStyle = "rgba(127,138,161,0.95)";
    ctx.font = "700 11px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`km/px = ${fmt(app.cam.kmPerPx, 3)}`, 10, 34);
    ctx.restore();
  }

  function hexToRgba(hex, a) {
    const h = hex.replace("#", "").trim();
    const r = parseInt(h.slice(0,2), 16);
    const g = parseInt(h.slice(2,4), 16);
    const b = parseInt(h.slice(4,6), 16);
    return `rgba(${r},${g},${b},${clamp(a,0,1)})`;
  }

  // -----------------------------
  // Plots
  // -----------------------------
  function drawPlotsIfNeeded() {
    const now = performance.now();
    if (!app.plotDirty && (now - app.lastPlotDraw) < 220) return;
    app.lastPlotDraw = now;
    app.plotDirty = false;

    drawStatesPlot();
    drawErrorsPlot();
    drawMeasPlot();
    drawConsPlot();
  }

  function drawStatesPlot() {
    const ctx = app.ctxStates;
    const canvas = el.plotStates;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    // If mobile: let JS draw a single column layout (2 panels tall * 2) by using a narrow tileW
    const isMobile = window.matchMedia && window.matchMedia("(max-width: 640px)").matches;

    const pad = 12;
    const cols = isMobile ? 1 : 2;
    const rows = isMobile ? 4 : 2;
    const tileW = (W - pad*(cols+1)) / cols;
    const tileH = (H - pad*(rows+1)) / rows;

    const labels = [
      { title:"X position", yLabel:"X (km)" },
      { title:"X velocity", yLabel:"Ẋ (km/s)" },
      { title:"Y position", yLabel:"Y (km)" },
      { title:"Y velocity", yLabel:"Ẏ (km/s)" },
    ];

    const tNow = app.t;
    app.viewStates.follow(tNow);
    const tr = app.viewStates.getRange(tNow);
    const tMin = tr.tMin, tMax = tr.tMax;

    for (let i = 0; i < 4; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = pad + col * (tileW + pad);
      const y = pad + row * (tileH + pad);

      const t = app.tHist;
      const truth = app.xTrueHist[i];
      const est = app.xHatHist[i];
      const sig = app.sigHist[i];
      if (!t.length) continue;

      const up = new Array(est.length);
      const lo = new Array(est.length);
      for (let k = 0; k < est.length; k++) {
        up[k] = est[k] + 2*sig[k];
        lo[k] = est[k] - 2*sig[k];
      }
      const { mn, mx } = computeMinMax([truth, est, up, lo]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: labels[i].title,
        xLabel: row === (rows-1) ? "t (s)" : "",
        yLabel: labels[i].yLabel,
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x + 8, py = y + 26, pw = tileW - 16, ph = tileH - 34;

      plotLine(ctx, px, py, pw, ph, t, truth, tMin, tMax, mn, mx, { stroke: "rgba(231,238,252,0.92)", width: 1.4 });
      plotLine(ctx, px, py, pw, ph, t, est,   tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.4 });
      plotLine(ctx, px, py, pw, ph, t, up,    tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.70)", width: 1.1, dash: [6,4] });
      plotLine(ctx, px, py, pw, ph, t, lo,    tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.70)", width: 1.1, dash: [6,4] });
    }
  }

  function drawErrorsPlot() {
    const ctx = app.ctxErrors;
    const canvas = el.plotErrors;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const isMobile = window.matchMedia && window.matchMedia("(max-width: 640px)").matches;
    const pad = 12;
    const rows = 4;
    const tileW = W - pad*2;
    const tileH = (H - pad*(rows+1)) / rows;

    const labels = [
      { title:"X position error", yLabel:"ΔX (km)" },
      { title:"X velocity error", yLabel:"ΔẊ (km/s)" },
      { title:"Y position error", yLabel:"ΔY (km)" },
      { title:"Y velocity error", yLabel:"ΔẎ (km/s)" },
    ];

    const tNow = app.t;
    app.viewErrors.follow(tNow);
    const tr = app.viewErrors.getRange(tNow);
    const tMin = tr.tMin, tMax = tr.tMax;

    for (let i = 0; i < 4; i++) {
      const x = pad;
      const y = pad + i*(tileH + pad);

      const t = app.tHist;
      const err = app.errHist[i];
      const sig = app.sigHist[i];
      if (!t.length) continue;

      const up = new Array(err.length);
      const lo = new Array(err.length);
      for (let k = 0; k < err.length; k++) {
        up[k] = 2*sig[k];
        lo[k] = -2*sig[k];
      }
      const { mn, mx } = computeMinMax([err, up, lo]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: labels[i].title,
        xLabel: (i === 3) ? "t (s)" : "",
        yLabel: labels[i].yLabel,
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x + 8, py = y + 26, pw = tileW - 16, ph = tileH - 34;

      plotLine(ctx, px, py, pw, ph, t, err, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.4 });
      plotLine(ctx, px, py, pw, ph, t, up,  tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });
      plotLine(ctx, px, py, pw, ph, t, lo,  tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });

      // zero line
      ctx.save();
      ctx.strokeStyle = "rgba(110,231,255,0.10)";
      ctx.lineWidth = 1;
      const y0 = py + ph * (1 - (0 - mn) / ((mx - mn) || 1));
      ctx.beginPath();
      ctx.moveTo(px, y0);
      ctx.lineTo(px + pw, y0);
      ctx.stroke();
      ctx.restore();

      // On very small screens, prioritize readability: fewer ticks
      if (isMobile) { /* tick density handled by drawAxesWithTicks defaults */ }
    }
  }

  function drawMeasPlot() {
    const ctx = app.ctxMeas;
    const canvas = el.plotMeas;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*5) / 3;

    const tNow = app.t;
    app.viewMeas.follow(tNow);
    const tr = app.viewMeas.getRange(tNow);
    const tMin = tr.tMin;
    const tMax = tr.tMax;

    // Gather combined series ranges across stations for y scaling
    const measRhoAll = [];
    const predRhoAll = [];
    const measRhodAll = [];
    const predRhodAll = [];
    const measPhiAll = [];
    const predPhiAll = [];

    for (let sid = 1; sid <= 12; sid++) {
      const m = app.measByStation[sid-1];
      const p = app.predByStation[sid-1];
      measRhoAll.push(...m.rho);
      predRhoAll.push(...p.rho);
      measRhodAll.push(...m.rhod);
      predRhodAll.push(...p.rhod);
      measPhiAll.push(...m.phi);
      predPhiAll.push(...p.phi);
    }

    // Range
    {
      const x = pad, y = pad;
      const { mn, mx } = computeMinMax([measRhoAll, predRhoAll]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: "Range ρ — measured vs predicted",
        xLabel: "",
        yLabel: "ρ (km)",
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x+8, py = y+26, pw = tileW-16, ph = tileH-34;
      for (let sid = 1; sid <= 12; sid++) {
        const color = stationColor(sid);
        const m = app.measByStation[sid-1];
        const p = app.predByStation[sid-1];

        plotScatter(ctx, px, py, pw, ph, m.t, m.rho, tMin, tMax, mn, mx, { fill: hexToRgba(color, 0.85), r: 2.0 });
        plotScatter(ctx, px, py, pw, ph, p.t, p.rho, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.70)", r: 1.8 });
      }
    }

    // Range-rate
    {
      const x = pad, y = pad + (tileH + pad);
      const { mn, mx } = computeMinMax([measRhodAll, predRhodAll]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: "Range-rate ρ̇ — measured vs predicted",
        xLabel: "",
        yLabel: "ρ̇ (km/s)",
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x+8, py = y+26, pw = tileW-16, ph = tileH-34;
      for (let sid = 1; sid <= 12; sid++) {
        const color = stationColor(sid);
        const m = app.measByStation[sid-1];
        const p = app.predByStation[sid-1];

        plotScatter(ctx, px, py, pw, ph, m.t, m.rhod, tMin, tMax, mn, mx, { fill: hexToRgba(color, 0.85), r: 2.0 });
        plotScatter(ctx, px, py, pw, ph, p.t, p.rhod, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.70)", r: 1.8 });
      }
    }

    // Bearing
    {
      const x = pad, y = pad + 2*(tileH + pad);
      const { mn, mx } = computeMinMax([measPhiAll, predPhiAll]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: "Bearing φ — measured vs predicted",
        xLabel: "t (s)",
        yLabel: "φ (rad)",
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x+8, py = y+26, pw = tileW-16, ph = tileH-34;
      for (let sid = 1; sid <= 12; sid++) {
        const color = stationColor(sid);
        const m = app.measByStation[sid-1];
        const p = app.predByStation[sid-1];

        plotScatter(ctx, px, py, pw, ph, m.t, m.phi, tMin, tMax, mn, mx, { fill: hexToRgba(color, 0.85), r: 2.0 });
        plotScatter(ctx, px, py, pw, ph, p.t, p.phi, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.70)", r: 1.8 });
      }
    }
  }

  function drawConsPlot() {
    const ctx = app.ctxCons;
    const canvas = el.plotCons;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*4) / 2;

    const alpha = clamp(parseFloat(el.inpAlpha?.value) || 0.05, 0.01, 0.2);

    const tNow = app.t;
    app.viewCons.follow(tNow);
    const tr = app.viewCons.getRange(tNow);
    const tMin = tr.tMin, tMax = tr.tMax;

    const t = app.tHist;

    // NEES df=4
    {
      const x = pad, y = pad;

      const df = 4;
      const lo = chi2Inv(alpha/2, df);
      const hi = chi2Inv(1 - alpha/2, df);
      const yLo = new Array(t.length).fill(lo);
      const yHi = new Array(t.length).fill(hi);

      const { mn, mx } = computeMinMax([app.neesHist, yLo, yHi]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: "NEES with χ² bounds",
        xLabel: "",
        yLabel: "NEES (—)",
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x+8, py = y+26, pw = tileW-16, ph = tileH-34;
      plotLine(ctx, px, py, pw, ph, t, app.neesHist, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.4 });
      plotLine(ctx, px, py, pw, ph, t, yLo,        tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });
      plotLine(ctx, px, py, pw, ph, t, yHi,        tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });
    }

    // NIS df=3*visible
    {
      const x = pad, y = pad + tileH + pad;

      const loArr = [];
      const hiArr = [];
      for (let i = 0; i < app.nisDfHist.length; i++) {
        const df = app.nisDfHist[i];
        if (!Number.isFinite(df) || df <= 0) { loArr.push(NaN); hiArr.push(NaN); continue; }
        loArr.push(chi2Inv(alpha/2, df));
        hiArr.push(chi2Inv(1 - alpha/2, df));
      }

      const { mn, mx } = computeMinMax([app.nisHist, loArr, hiArr]);

      drawAxesWithTicks(ctx, x, y, tileW, tileH, {
        title: "NIS with χ² bounds",
        xLabel: "t (s)",
        yLabel: "NIS (—)",
        xMin: tMin, xMax: tMax,
        yMin: mn, yMax: mx
      });

      const px = x+8, py = y+26, pw = tileW-16, ph = tileH-34;
      plotLine(ctx, px, py, pw, ph, t, app.nisHist, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.4 });
      plotLine(ctx, px, py, pw, ph, t, loArr,      tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });
      plotLine(ctx, px, py, pw, ph, t, hiArr,      tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.1, dash: [6,4] });
    }
  }

  // -----------------------------
  // Orbit canvas pan/zoom
  // -----------------------------
  function getPointerXY(e) {
    if (e.touches && e.touches.length) return { x: e.touches[0].clientX, y: e.touches[0].clientY };
    return { x: e.clientX, y: e.clientY };
  }

  function setupOrbitPanZoom() {
    const canvas = el.orbitCanvas;
    if (!canvas) return;

    let panning = false;
    let last = { x: 0, y: 0 };
    let pinch = { active: false, idA: null, idB: null, dist0: 0, kmPerPx0: 0, mid0: null };
    const pointers = new Map();

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.size === 1) {
        panning = true;
        last = { x: e.clientX, y: e.clientY };
      } else if (pointers.size === 2) {
        const ids = Array.from(pointers.keys());
        const A = pointers.get(ids[0]);
        const B = pointers.get(ids[1]);
        pinch.active = true;
        pinch.idA = ids[0];
        pinch.idB = ids[1];
        pinch.dist0 = Math.hypot(B.x - A.x, B.y - A.y);
        pinch.kmPerPx0 = app.cam.kmPerPx;

        const rect = canvas.getBoundingClientRect();
        pinch.mid0 = { x: (A.x + B.x)/2 - rect.left, y: (A.y + B.y)/2 - rect.top };
        panning = false;
      }

      e.preventDefault();
    }, { passive: false });

    canvas.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      const rect = canvas.getBoundingClientRect();
      const W = rect.width, H = rect.height;

      if (pinch.active && pointers.size === 2) {
        const A = pointers.get(pinch.idA);
        const B = pointers.get(pinch.idB);
        if (!A || !B) return;

        const dist = Math.hypot(B.x - A.x, B.y - A.y);
        const scale = dist / (pinch.dist0 + 1e-9);
        const newKmPerPx = clamp(pinch.kmPerPx0 / scale, 2, 500);

        const mid = pinch.mid0;
        const before = app.cam.screenToWorld(mid.x, mid.y, W, H);
        app.cam.kmPerPx = newKmPerPx;
        const after = app.cam.screenToWorld(mid.x, mid.y, W, H);
        app.cam.cx += (before.x - after.x);
        app.cam.cy += (before.y - after.y);

        e.preventDefault();
        return;
      }

      if (!panning) return;

      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };

      app.cam.cx -= dx * app.cam.kmPerPx;
      app.cam.cy -= dy * app.cam.kmPerPx;

      e.preventDefault();
    }, { passive: false });

    function endPointer(e) {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch.active = false;
      if (pointers.size === 0) panning = false;
    }
    canvas.addEventListener("pointerup", endPointer, { passive: true });
    canvas.addEventListener("pointercancel", endPointer, { passive: true });

    canvas.addEventListener("wheel", (e) => {
      const rect = canvas.getBoundingClientRect();
      const W = rect.width, H = rect.height;

      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const before = app.cam.screenToWorld(mx, my, W, H);
      const zoom = Math.exp(-e.deltaY * 0.0015);
      app.cam.kmPerPx = clamp(app.cam.kmPerPx / zoom, 2, 500);
      const after = app.cam.screenToWorld(mx, my, W, H);

      app.cam.cx += (before.x - after.x);
      app.cam.cy += (before.y - after.y);
      e.preventDefault();
    }, { passive: false });

    if (el.btnResetView) el.btnResetView.addEventListener("click", () => app.cam.reset(app.Re));
    if (el.btnCenterView) el.btnCenterView.addEventListener("click", () => { app.cam.cx = 0; app.cam.cy = 0; });
    if (el.btnZoomToSat) el.btnZoomToSat.addEventListener("click", () => {
      // zoom in tight around spacecraft to highlight measurement effects
      app.cam.cx = app.truth[0];
      app.cam.cy = app.truth[2];
      app.cam.kmPerPx = clamp(app.Re / 900, 2, 30);
    });
  }

  // -----------------------------
  // Time-window pan/zoom on plots (paused only)
  // -----------------------------
  function attachPlotPanZoom(canvas, view) {
    if (!canvas) return;
    let dragging = false;
    let lastX = 0;

    canvas.addEventListener("pointerdown", (e) => {
      if (app.running) return;
      dragging = true;
      view.userTouched();
      lastX = e.clientX;
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    }, { passive:false });

    canvas.addEventListener("pointermove", (e) => {
      if (app.running || !dragging) return;
      const dx = e.clientX - lastX;
      lastX = e.clientX;
      // pan in time: pixels -> seconds (approx)
      const rect = canvas.getBoundingClientRect();
      const secPerPx = view.windowSec / Math.max(50, rect.width);
      view.pan(-dx * secPerPx);
      app.plotDirty = true;
      e.preventDefault();
    }, { passive:false });

    canvas.addEventListener("pointerup", () => { dragging = false; }, { passive:true });
    canvas.addEventListener("pointercancel", () => { dragging = false; }, { passive:true });

    canvas.addEventListener("wheel", (e) => {
      if (app.running) return;
      view.userTouched();
      const zoom = Math.exp(-e.deltaY * 0.0012);
      view.zoom(zoom);
      app.plotDirty = true;
      e.preventDefault();
    }, { passive:false });
  }

  // -----------------------------
  // Thrust control (“joystick” buttons + WASD)
  // -----------------------------
  function readThrustMagKmps2() {
    // Prefer m/s^2 input if present, convert to km/s^2
    if (el.inpThrustMagMS2 && el.inpThrustMagMS2.value !== "") {
      const a = parseFloat(el.inpThrustMagMS2.value);
      if (Number.isFinite(a)) return clamp(a, 0, 0.2) / 1000; // m/s^2 -> km/s^2
    }
    if (el.inpThrustMag && el.inpThrustMag.value !== "") {
      const a = parseFloat(el.inpThrustMag.value);
      if (Number.isFinite(a)) return clamp(a, 0, 2e-4); // km/s^2
    }
    // fallback: mild constant
    return 0.0;
  }

  function updateThrustCommand() {
    app.thrustMag = readThrustMagKmps2();

    let ax = 0, ay = 0;
    if (app.uHold.up) ay += 1;
    if (app.uHold.down) ay -= 1;
    if (app.uHold.right) ax += 1;
    if (app.uHold.left) ax -= 1;

    // normalize diagonals
    const n = Math.hypot(ax, ay);
    if (n > 1e-9) { ax /= n; ay /= n; }

    app.uCmd[0] = ax * app.thrustMag;
    app.uCmd[1] = ay * app.thrustMag;
  }

  function bindHoldButton(btn, dirKey) {
    if (!btn) return;
    const down = () => { app.uHold[dirKey] = true; updateThrustCommand(); };
    const up = () => { app.uHold[dirKey] = false; updateThrustCommand(); };

    btn.addEventListener("pointerdown", (e) => { btn.setPointerCapture(e.pointerId); down(); e.preventDefault(); }, { passive:false });
    btn.addEventListener("pointerup", (e) => { up(); e.preventDefault(); }, { passive:false });
    btn.addEventListener("pointercancel", () => { up(); }, { passive:true });
    btn.addEventListener("pointerleave", () => { /* leave doesn’t necessarily cancel capture */ }, { passive:true });
  }

  function setupThrustUI() {
    bindHoldButton(el.thrUp, "up");
    bindHoldButton(el.thrDown, "down");
    bindHoldButton(el.thrLeft, "left");
    bindHoldButton(el.thrRight, "right");

    if (el.thrZero) {
      el.thrZero.addEventListener("click", () => {
        app.uHold.up = app.uHold.down = app.uHold.left = app.uHold.right = false;
        updateThrustCommand();
      });
    }

    window.addEventListener("keydown", (e) => {
      const k = e.key.toLowerCase();
      if (k === "w") app.uHold.up = true;
      if (k === "s") app.uHold.down = true;
      if (k === "a") app.uHold.left = true;
      if (k === "d") app.uHold.right = true;
      if (k === " ") { app.uHold.up = app.uHold.down = app.uHold.left = app.uHold.right = false; }
      updateThrustCommand();
    });

    window.addEventListener("keyup", (e) => {
      const k = e.key.toLowerCase();
      if (k === "w") app.uHold.up = false;
      if (k === "s") app.uHold.down = false;
      if (k === "a") app.uHold.left = false;
      if (k === "d") app.uHold.right = false;
      updateThrustCommand();
    });
  }

  // -----------------------------
  // Visual link events
  // -----------------------------
  function spawnSignalPulses(xTrue, tNow, stationIds) {
    for (const id of stationIds) {
      const s = stationState(id - 1, tNow, app.Re, app.omega);
      app.pulses.push({
        t0: performance.now(),
        dur: 360,
        sx: xTrue[0],
        sy: xTrue[2],
        tx: s.X,
        ty: s.Y,
        id,
        color: stationColor(id),
      });
    }
    if (app.pulses.length > 240) app.pulses.splice(0, app.pulses.length - 240);
  }

  // -----------------------------
  // Simulation step
  // -----------------------------
  function wouldImpactEarth(nextX) {
    const r = Math.hypot(nextX[0], nextX[2]);
    return r <= (CONST.Re + 1.0);
  }

  function captureSafeState() {
    app.lastSafeState = {
      t: app.t,
      k: app.k,
      truth: new Float64Array(app.truth),
      ekfX: new Float64Array(app.ekf.x),
      ekfP: matCopy(app.ekf.P),
    };
  }

  function restoreSafeState() {
    if (!app.lastSafeState) return;
    app.t = app.lastSafeState.t;
    app.k = app.lastSafeState.k;
    app.truth = new Float64Array(app.lastSafeState.truth);
    app.ekf.x = new Float64Array(app.lastSafeState.ekfX);
    app.ekf.P = matCopy(app.lastSafeState.ekfP);
  }

  function simulateOneStep() {
    if (app.k >= app.Kmax) {
      setRunning(false);
      return;
    }

    // Live toggles / noise params (if present)
    app.enableProcNoise = !!(el.tglProcNoise?.checked ?? true);
    app.enableMeasNoise = !!(el.tglMeasNoise?.checked ?? true);

    // True noise (slightly adjustable)
    if (el.inpQtrueAx || el.inpQtrueAy) {
      const qx = parseFloat(el.inpQtrueAx?.value) || app.QtrueAcc[0][0];
      const qy = parseFloat(el.inpQtrueAy?.value) || app.QtrueAcc[1][1];
      app.QtrueAcc = buildQacc(Math.max(0, qx), Math.max(0, qy));
    }
    if (el.inpRtrueRho || el.inpRtrueRhod || el.inpRtruePhi) {
      const sr = parseFloat(el.inpRtrueRho?.value) || 5;
      const sv = parseFloat(el.inpRtrueRhod?.value) || 0.01;
      const sa = parseFloat(el.inpRtruePhi?.value) || 0.01;
      app.RtrueStn = buildRstn(Math.max(0, sr), Math.max(0, sv), Math.max(0, sa));
    }

    // EKF noise
    const qkfAx = parseFloat(el.inpQkfAx?.value) || 1e-8;
    const qkfAy = parseFloat(el.inpQkfAy?.value) || 1e-8;
    const rkfRho = parseFloat(el.inpRkfRho?.value) || 5;
    const rkfRhod = parseFloat(el.inpRkfRhod?.value) || 0.01;
    const rkfPhi = parseFloat(el.inpRkfPhi?.value) || 0.01;

    app.ekf.setQR(
      buildQacc(Math.max(0, qkfAx), Math.max(0, qkfAy)),
      buildRstn(Math.max(0, rkfRho), Math.max(0, rkfRhod), Math.max(0, rkfPhi))
    );

    // Force command update
    updateThrustCommand();

    const dt = app.dt;
    const tNext = app.t + dt;

    // Save a safe state occasionally for recovery
    if (app.k % 5 === 0) captureSafeState();

    // Truth propagation
    let xTrueNext = rk4Step(app.truth, app.uCmd, dt, app.mu);

    // If thrust pushes into Earth, stop and recover
    if (wouldImpactEarth(xTrueNext)) {
      setRunning(false);
      restoreSafeState();
      // Clear thrust latch (prevents repeated impact)
      app.uHold.up = app.uHold.down = app.uHold.left = app.uHold.right = false;
      updateThrustCommand();
      app.plotDirty = true;
      return;
    }

    if (app.enableProcNoise) {
      const QdTrue = QdFromQacc(app.QtrueAcc, dt);
      const w = sampleMVN(4, QdTrue, app.randn);
      for (let i = 0; i < 4; i++) xTrueNext[i] += w[i];
    }

    // Visible stations at tNext
    const vis = visibleStations(xTrueNext, tNext, app.Re, app.omega);

    // Measurements (stacked)
    let yMeas = new Float64Array(0);
    if (vis.length > 0) {
      yMeas = new Float64Array(3 * vis.length);
      for (let k = 0; k < vis.length; k++) {
        const sid = vis[k];
        const m = measureOneStation(xTrueNext, tNext, sid, app.Re, app.omega);
        yMeas[3*k+0] = m.rho;
        yMeas[3*k+1] = m.rhod;
        yMeas[3*k+2] = m.phi;
      }

      if (app.enableMeasNoise) {
        const mDim = yMeas.length;
        const R = zeros(mDim, mDim);
        for (let k = 0; k < vis.length; k++) {
          for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
            R[3*k+i][3*k+j] = app.RtrueStn[i][j];
          }
        }
        const v = sampleMVN(mDim, R, app.randn);
        for (let i = 0; i < mDim; i++) yMeas[i] += v[i];
      }

      // Store colored measurements by station
      for (let k = 0; k < vis.length; k++) {
        const sid = vis[k];
        const bin = app.measByStation[sid-1];
        pushLimited(bin.t, tNext);
        pushLimited(bin.rho,  yMeas[3*k+0]);
        pushLimited(bin.rhod, yMeas[3*k+1]);
        pushLimited(bin.phi,  yMeas[3*k+2]);
      }
    }

    // EKF update
    app.ekf.step(dt, tNext, app.uCmd, yMeas, vis);

    // Predicted measurement storage per station
    if (app.ekf.lastYhat && app.ekf.lastStationIds && app.ekf.lastStationIds.length > 0) {
      const yhat = app.ekf.lastYhat;
      const ids = app.ekf.lastStationIds;
      for (let k = 0; k < ids.length; k++) {
        const sid = ids[k];
        const bin = app.predByStation[sid-1];
        pushLimited(bin.t, tNext);
        pushLimited(bin.rho,  yhat[3*k+0]);
        pushLimited(bin.rhod, yhat[3*k+1]);
        pushLimited(bin.phi,  yhat[3*k+2]);
      }
    }

    // pulses (visual event)
    if (vis.length > 0) spawnSignalPulses(xTrueNext, tNext, vis);

    // Advance
    app.truth = xTrueNext;
    app.t = tNext;
    app.k += 1;

    pushHistorySample();
    updateReadouts(vis);
    updateStationBoard(vis, vis);
  }

  // -----------------------------
  // Reset / initialization
  // -----------------------------
  function hardResetAll() {
    app.dt = CONST.dt;
    app.mu = CONST.mu;
    app.Re = CONST.Re;
    app.omega = CONST.omega;

    // steps
    if (el.inpSteps) {
      const kmax = parseInt(el.inpSteps.value || "1000000", 10);
      app.Kmax = clamp(kmax, 500, 10_000_000);
    } else {
      app.Kmax = 1_000_000;
    }

    // random seed every time (no user seed)
    const seed = (Math.random() * 0xFFFFFFFF) >>> 0;
    app.rng = mulberry32(seed);
    app.randn = makeNormal(app.rng);

    // set initial orbit from elements (with defaults if inputs absent)
    const rpAlt = parseFloat(el.inpRpAlt?.value) || 350;
    const ecc = clamp(parseFloat(el.inpEcc?.value) || 0.02, 0, 0.8);
    const nuDeg = parseFloat(el.inpNuDeg?.value) || 0;
    const argPerDeg = parseFloat(el.inpArgPerDeg?.value) || 0;

    const x0True = elementsToState2D(rpAlt, ecc, nuDeg, argPerDeg);

    // EKF initial uncertainty + small bias
    const sigX0 = parseFloat(el.inpSigX0?.value) || 5;     // km
    const sigV0 = parseFloat(el.inpSigV0?.value) || 0.02;  // km/s
    const P0 = buildP0(sigX0, sigV0);
    const x0Hat = new Float64Array([
      x0True[0] + sigX0,
      x0True[1] + sigV0,
      x0True[2] + sigX0,
      x0True[3] + sigV0
    ]);

    app.ekf = new EKF(app.mu, app.Re, app.omega);
    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx?.value) || 1e-8, parseFloat(el.inpQkfAy?.value) || 1e-8),
      buildRstn(parseFloat(el.inpRkfRho?.value) || 5, parseFloat(el.inpRkfRhod?.value) || 0.01, parseFloat(el.inpRkfPhi?.value) || 0.01)
    );
    app.ekf.reset(x0Hat, P0);

    // clear thrust
    app.uHold.up = app.uHold.down = app.uHold.left = app.uHold.right = false;
    updateThrustCommand();

    // reset time
    app.t = 0;
    app.k = 0;
    app.truth = new Float64Array(x0True);

    // clear histories
    app.tHist = [];
    app.xTrueHist = [[],[],[],[]];
    app.xHatHist  = [[],[],[],[]];
    app.sigHist   = [[],[],[],[]];
    app.errHist   = [[],[],[],[]];
    app.neesHist = [];
    app.nisHist = [];
    app.nisDfHist = [];

    app.measByStation = Array.from({length:12}, () => ({ t:[], rho:[], rhod:[], phi:[] }));
    app.predByStation = Array.from({length:12}, () => ({ t:[], rho:[], rhod:[], phi:[] }));

    app.pulses = [];
    app.plotDirty = true;

    app.cam.reset(app.Re);

    // safe state seed
    app.lastSafeState = null;
    captureSafeState();

    setRunning(false);
    pushHistorySample();
    updateReadouts([]);
    updateStationBoard([], []);
  }

  function resetEKFOnly() {
    const sigX0 = parseFloat(el.inpSigX0?.value) || 5;
    const sigV0 = parseFloat(el.inpSigV0?.value) || 0.02;
    const P0 = buildP0(sigX0, sigV0);
    const xTrue = app.truth;
    const x0Hat = new Float64Array([xTrue[0] + sigX0, xTrue[1] + sigV0, xTrue[2] + sigX0, xTrue[3] + sigV0]);

    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx?.value) || 1e-8, parseFloat(el.inpQkfAy?.value) || 1e-8),
      buildRstn(parseFloat(el.inpRkfRho?.value) || 5, parseFloat(el.inpRkfRhod?.value) || 0.01, parseFloat(el.inpRkfPhi?.value) || 0.01)
    );
    app.ekf.reset(x0Hat, P0);

    // reset EKF histories (keep truth history)
    app.xHatHist = [[], [], [], []];
    app.sigHist = [[], [], [], []];
    app.errHist = [[], [], [], []];
    app.neesHist = [];
    app.nisHist = [];
    app.nisDfHist = [];

    app.predByStation = Array.from({length:12}, () => ({ t:[], rho:[], rhod:[], phi:[] }));

    // align arrays with existing time history using NaNs
    for (let idx = 0; idx < app.tHist.length; idx++) {
      for (let j = 0; j < 4; j++) {
        app.xHatHist[j].push(NaN);
        app.sigHist[j].push(NaN);
        app.errHist[j].push(NaN);
      }
      app.neesHist.push(NaN);
      app.nisHist.push(NaN);
      app.nisDfHist.push(NaN);
    }

    app.plotDirty = true;
  }

  // -----------------------------
  // UI wiring
  // -----------------------------
  function setupSpeedButtons() {
    const group = document.querySelectorAll(".chip[data-speed]");
    group.forEach((btn) => btn.addEventListener("click", () => setSpeed(parseFloat(btn.dataset.speed))));
  }

  function setupControlButtons() {
    if (el.btnToggleRun) el.btnToggleRun.addEventListener("click", () => setRunning(!app.running));
    if (el.btnStep) el.btnStep.addEventListener("click", () => { if (!app.running) simulateOneStep(); });
    if (el.btnReset) el.btnReset.addEventListener("click", hardResetAll);
    if (el.btnApplyIC) el.btnApplyIC.addEventListener("click", hardResetAll);
    if (el.btnResetFilter) el.btnResetFilter.addEventListener("click", resetEKFOnly);
  }

  function enforceNonTunableConstants() {
    // If older inputs exist in the DOM, lock them.
    const lockIds = ["inpDt","inpMu","inpRe","inpOmega","inpSeed"];
    for (const id of lockIds) {
      const node = $(`#${id}`);
      if (!node) continue;
      node.value = (id === "inpDt") ? String(CONST.dt)
              : (id === "inpMu") ? String(CONST.mu)
              : (id === "inpRe") ? String(CONST.Re)
              : (id === "inpOmega") ? String(CONST.omega)
              : "";
      node.disabled = true;
      node.setAttribute("aria-disabled", "true");
    }

    // Ensure steps defaults big if present
    if (el.inpSteps && (!el.inpSteps.value || parseInt(el.inpSteps.value,10) < 1000000)) {
      el.inpSteps.value = "1000000";
    }
  }

  // -----------------------------
  // Resize handling
  // -----------------------------
  function handleResize(force = false) {
    app.orbitCtx  = setupCanvasHiDPI(el.orbitCanvas);
    app.ctxStates = setupCanvasHiDPI(el.plotStates);
    app.ctxErrors = setupCanvasHiDPI(el.plotErrors);
    app.ctxMeas   = setupCanvasHiDPI(el.plotMeas);
    app.ctxCons   = setupCanvasHiDPI(el.plotCons);

    app.plotDirty = true;
    if (force) {
      drawOrbitView();
      drawPlotsIfNeeded();
    }
  }

  // -----------------------------
  // Animation loop (speed maps sim time to wall time)
  // -----------------------------
  let lastRAF = 0;
  let simTimeAccumulator = 0;

  function tickRAF(ts) {
    if (!lastRAF) lastRAF = ts;
    const dtReal = (ts - lastRAF) / 1000;
    lastRAF = ts;

    if (app.running) {
      simTimeAccumulator += dtReal * app.speed;

      const maxStepsPerFrame = 60;
      let steps = 0;
      while (simTimeAccumulator >= app.dt && steps < maxStepsPerFrame) {
        simulateOneStep();
        simTimeAccumulator -= app.dt;
        steps++;
      }
      if (steps === maxStepsPerFrame) simTimeAccumulator = 0;
    } else {
      simTimeAccumulator = 0;
    }

    drawOrbitView();
    drawPlotsIfNeeded();
    requestAnimationFrame(tickRAF);
  }

  // -----------------------------
  // Init
  // -----------------------------
  function init() {
    buildStationBoard();
    setupSpeedButtons();
    setupControlButtons();
    setupOrbitPanZoom();
    setupThrustUI();
    enforceNonTunableConstants();

    // plot pan/zoom when paused
    attachPlotPanZoom(el.plotMeas, app.viewMeas);
    attachPlotPanZoom(el.plotCons, app.viewCons);
    attachPlotPanZoom(el.plotStates, app.viewStates);
    attachPlotPanZoom(el.plotErrors, app.viewErrors);

    // default speed
    setSpeed(10);

    handleResize(true);
    window.addEventListener("resize", () => handleResize(true));

    hardResetAll();
    requestAnimationFrame(tickRAF);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
