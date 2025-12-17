/* orbit-sim.js
   Client-side orbit + EKF simulator (2D).
   No external libs, no network calls, no eval.

   This file assumes index.html provides the UI elements by id/class.
   If an element is missing (e.g., you rename an id), the simulator will
   continue running; that feature just becomes inactive until the id matches.
*/
(() => {
  "use strict";

  // -----------------------------
  // DOM helpers
  // -----------------------------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

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

  function safeText(el, s) { if (el) el.textContent = s; }
  function safeToggle(el, cls, on) { if (el) el.classList.toggle(cls, !!on); }

  // -----------------------------
  // RNG: Mulberry32 + Box-Muller
  // -----------------------------
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
  // Small linear algebra (dense)
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
      if (best < 1e-18) throw new Error("Singular matrix in inversion.");
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
  // Statistics: inverse normal + chi-square approx
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
  // Dynamics + measurement models
  // -----------------------------
  function orbitDeriv(x, u, mu) {
    // x = [X, Xdot, Y, Ydot] in km, km/s
    // u = [ax, ay] in km/s^2
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
    // 12 stations equally spaced around Earth in inertial frame
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
      if (dot > 0) vis.push(i+1); // station ids 1..12
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

    return { rho, rhod, phi, dX, dY, dXd, dYd, st: s };
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

      H[3*k + 0][0] = m.dX / rho;
      H[3*k + 0][2] = m.dY / rho;

      H[3*k + 1][0] = (m.dXd / rho) - (a * m.dX / rho3);
      H[3*k + 1][2] = (m.dYd / rho) - (a * m.dY / rho3);
      H[3*k + 1][1] = m.dX / rho;
      H[3*k + 1][3] = m.dY / rho;

      const rho2 = rho*rho;
      H[3*k + 2][0] = -m.dY / rho2;
      H[3*k + 2][2] =  m.dX / rho2;
    }
    return { yhat, H };
  }

  // Discrete-time mapping for accel noise -> state noise
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

      if (!yMeas || yMeas.length === 0 || !stationIds || stationIds.length === 0) {
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
      const S = matAdd(matMul(HP, matTrans(H)), R);
      const Ssym = symmetrize(S);

      const PHt = matMul(Pminus, matTrans(H));
      const Sinv = matInv(Ssym);
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
      this.lastS = Ssym;
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
  // HiDPI canvas setup
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

  // -----------------------------
  // Plot primitives (clip-safe)
  // -----------------------------
  function clearPanel(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const g = ctx.createRadialGradient(w*0.35, h*0.25, 8, w*0.5, h*0.5, Math.max(w,h)*0.85);
    g.addColorStop(0, "rgba(59,130,246,0.10)");  // blue
    g.addColorStop(1, "rgba(249,115,22,0.06)");  // orange
    ctx.fillStyle = g;
    ctx.fillRect(0,0,w,h);
  }

  function drawAxes(ctx, x, y, w, h, opts) {
    const {
      grid = true,
      title = "",
      xLabel = "",
      yLabel = "",
      xTicks = null, // {min,max,unit,fmt}
      yTicks = null
    } = opts || {};

    ctx.save();
    ctx.translate(x,y);

    // frame
    ctx.strokeStyle = "rgba(226,232,240,1)"; // slate-200
    ctx.lineWidth = 1;
    ctx.strokeRect(0,0,w,h);

    // title
    ctx.fillStyle = "rgba(15,23,42,0.95)"; // slate-900
    ctx.font = "900 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(title, 10, 16);

    // labels
    ctx.fillStyle = "rgba(71,85,105,0.95)"; // slate-600
    ctx.font = "800 11px ui-monospace, Menlo, Consolas, monospace";
    if (xLabel) ctx.fillText(xLabel, w - ctx.measureText(xLabel).width - 10, h - 8);

    if (yLabel) {
      ctx.save();
      ctx.translate(12, h/2);
      ctx.rotate(-Math.PI/2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    // grid + ticks
    const plotX0 = 46;
    const plotY0 = 26;
    const plotW = Math.max(2, w - plotX0 - 10);
    const plotH = Math.max(2, h - plotY0 - 22);

    if (grid) {
      ctx.strokeStyle = "rgba(226,232,240,0.85)";
      ctx.lineWidth = 1;
      const nx = 6, ny = 4;

      for (let i = 0; i <= nx; i++) {
        const gx = plotX0 + (i/nx)*plotW;
        ctx.beginPath();
        ctx.moveTo(gx, plotY0);
        ctx.lineTo(gx, plotY0 + plotH);
        ctx.stroke();
      }
      for (let j = 0; j <= ny; j++) {
        const gy = plotY0 + (j/ny)*plotH;
        ctx.beginPath();
        ctx.moveTo(plotX0, gy);
        ctx.lineTo(plotX0 + plotW, gy);
        ctx.stroke();
      }

      // ticks text
      ctx.fillStyle = "rgba(100,116,139,0.95)"; // slate-500
      ctx.font = "800 10px ui-monospace, Menlo, Consolas, monospace";

      if (xTicks && Number.isFinite(xTicks.min) && Number.isFinite(xTicks.max) && xTicks.max > xTicks.min) {
        for (let i = 0; i <= nx; i++) {
          const v = xTicks.min + (i/nx)*(xTicks.max - xTicks.min);
          const s = (xTicks.fmt ? xTicks.fmt(v) : fmt(v, 0)) + (xTicks.unit ? ` ${xTicks.unit}` : "");
          const gx = plotX0 + (i/nx)*plotW;
          ctx.fillText(s, gx - ctx.measureText(s).width/2, plotY0 + plotH + 14);
        }
      }

      if (yTicks && Number.isFinite(yTicks.min) && Number.isFinite(yTicks.max) && yTicks.max > yTicks.min) {
        for (let j = 0; j <= ny; j++) {
          const v = yTicks.max - (j/ny)*(yTicks.max - yTicks.min);
          const s = (yTicks.fmt ? yTicks.fmt(v) : fmt(v, 2)) + (yTicks.unit ? ` ${yTicks.unit}` : "");
          const gy = plotY0 + (j/ny)*plotH;
          ctx.fillText(s, 6, gy + 3);
        }
      }
    }

    ctx.restore();
    return { plotX0: x + plotX0, plotY0: y + plotY0, plotW, plotH };
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

  function plotLine(ctx, box, t, y, tMin, tMax, yMin, yMax, style) {
    if (!t || !y || t.length < 2) return;
    const n = Math.min(t.length, y.length);
    const sx = (tt) => box.x + (tt - tMin) / (tMax - tMin) * box.w;
    const sy = (vv) => box.y + box.h - (vv - yMin) / (yMax - yMin) * box.h;

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();

    ctx.strokeStyle = style.stroke || "rgba(37,99,235,0.9)"; // blue-600
    ctx.lineWidth = style.width || 1.5;
    if (style.dash) ctx.setLineDash(style.dash);

    let started = false;
    for (let i = 0; i < n; i++) {
      const tt = t[i], vv = y[i];
      if (!Number.isFinite(tt) || !Number.isFinite(vv)) continue;
      const px = sx(tt), py = sy(vv);
      if (!started) { ctx.beginPath(); ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    if (started) ctx.stroke();
    ctx.restore();
  }

  function plotScatter(ctx, box, t, y, tMin, tMax, yMin, yMax, style, colors = null) {
    if (!t || !y || t.length < 1) return;
    const n = Math.min(t.length, y.length);
    const sx = (tt) => box.x + (tt - tMin) / (tMax - tMin) * box.w;
    const sy = (vv) => box.y + box.h - (vv - yMin) / (yMax - yMin) * box.h;

    const r = style.r || 2.2;

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();

    for (let i = 0; i < n; i++) {
      const tt = t[i], vv = y[i];
      if (!Number.isFinite(tt) || !Number.isFinite(vv)) continue;
      const px = sx(tt), py = sy(vv);

      ctx.fillStyle = (colors && colors[i]) ? colors[i] : (style.fill || "rgba(249,115,22,0.85)");
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2*Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // -----------------------------
  // Orbit camera + interactions
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
  // Station colors (12 distinct)
  // -----------------------------
  const ST_COLORS = [
    "#2563eb", "#ea580c", "#16a34a", "#dc2626",
    "#7c3aed", "#0891b2", "#d97706", "#0f766e",
    "#be123c", "#1d4ed8", "#65a30d", "#b45309"
  ];

  // -----------------------------
  // App DOM handles (safe)
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
    stationBoardSmall: $("#stationBoardSmall"), // optional (meas sidebar)

    btnToggleRun: $("#btnToggleRun"),
    btnStep: $("#btnStep"),
    btnReset: $("#btnReset"),
    btnApplyIC: $("#btnApplyIC"),
    btnResetFilter: $("#btnResetFilter"),

    inpDt: $("#inpDt"),
    inpSteps: $("#inpSteps"),

    tglProcNoise: $("#tglProcNoise"),
    tglMeasNoise: $("#tglMeasNoise"),

    // Truth noise
    inpQtrueAx: $("#inpQtrueAx"),
    inpQtrueAy: $("#inpQtrueAy"),
    inpRtrueRho: $("#inpRtrueRho"),
    inpRtrueRhod: $("#inpRtrueRhod"),
    inpRtruePhi: $("#inpRtruePhi"),

    // Filter assumptions
    inpSigX0: $("#inpSigX0"),
    inpSigV0: $("#inpSigV0"),
    inpQkfAx: $("#inpQkfAx"),
    inpQkfAy: $("#inpQkfAy"),
    inpRkfRho: $("#inpRkfRho"),
    inpRkfRhod: $("#inpRkfRhod"),
    inpRkfPhi: $("#inpRkfPhi"),
    inpAlpha: $("#inpAlpha"),

    // Thrust controls (optional; ids may vary)
    inpThrust: $("#inpThrust"),
    btnThUp: $("#btnThUp"),
    btnThDown: $("#btnThDown"),
    btnThLeft: $("#btnThLeft"),
    btnThRight: $("#btnThRight"),
    btnThZero: $("#btnThZero"),

    // Speed chips
    spdChips: $$(".chip[data-speed]"),

    // Orbit view
    orbitCanvas: $("#orbitCanvas"),
    btnCenterView: $("#btnCenterView"),
    btnResetView: $("#btnResetView"),
    btnZoomCraft: $("#btnZoomCraft"),

    // Plots
    plotStates: $("#plotStates"),
    plotErrors: $("#plotErrors"),
    plotMeas: $("#plotMeas"),
    plotTraj: $("#plotTraj"),
    plotCons: $("#plotCons"),

    // Tabs
    tabs: $$(".tab"),
    paneStates: $("#pane-states"),
    paneErrors: $("#pane-errors"),
    paneMeas: $("#pane-meas"),
    paneTraj: $("#pane-traj"),
    paneCons: $("#pane-cons"),
  };

  // -----------------------------
  // App state
  // -----------------------------
  const app = {
    running: false,
    speed: 10,
    dt: 10,
    Kmax: 1_000_000,

    // Earth constants (fixed)
    mu: 398600,             // km^3/s^2
    Re: 6378,               // km
    omega: 2*Math.PI/86400, // rad/s

    t: 0,
    k: 0,
    truth: new Float64Array(4),
    ekf: new EKF(398600, 6378, 2*Math.PI/86400),

    enableProcNoise: true,
    enableMeasNoise: true,

    QtrueAcc: eye(2),
    RtrueStn: eye(3),

    // control acceleration (km/s^2)
    u: new Float64Array([0,0]),
    thrustMag: 0.00002, // default tiny accel

    // history buffers
    bufMax: 2400,
    tHist: [],
    xTrueHist: [[],[],[],[]],
    xHatHist: [[],[],[],[]],
    sigHist: [[],[],[],[]],
    errHist: [[],[],[],[]],
    neesHist: [],
    nisHist: [],
    nisDfHist: [],

    // measurement series (store station ids for coloring)
    measT: [],
    measRho: [],
    measRhod: [],
    measPhi: [],
    measStId: [],

    predT: [],
    predRho: [],
    predRhod: [],
    predPhi: [],
    predStId: [],

    // signal visuals
    pulses: [],     // line + dot
    squiggles: [],  // animated squiggle along link

    // render handles
    cam: new Camera(),
    orbitCtx: null,
    ctxStates: null,
    ctxErrors: null,
    ctxMeas: null,
    ctxTraj: null,
    ctxCons: null,

    // plotting
    plotDirty: true,
    lastPlotDraw: 0,

    // RNG (random each load)
    rng: null,
    randn: null,

    // thrust input state
    thrustKeys: { up:false, down:false, left:false, right:false },
    thrustButtonsHeld: { up:false, down:false, left:false, right:false },

    // crash flag
    crashed: false,
  };

  // -----------------------------
  // Station board UI
  // -----------------------------
  function makeDishSVG() {
    // Dish is oriented "up" (bowl opening upward). The earlier inverted version looked upside down.
    return `
      <svg viewBox="0 0 120 34" aria-hidden="true" focusable="false">
        <path class="dishfill" d="M 12 18 Q 60 2 108 18 Q 60 26 12 18 Z"></path>
        <path class="dishline" d="M 12 18 Q 60 2 108 18"></path>
        <path class="dishline" d="M 58 18 L 50 7"></path>
        <path class="dishline" d="M 60 18 L 60 6"></path>
        <path class="dishline" d="M 62 18 L 70 7"></path>
        <path class="dishline" d="M 54 18 L 54 31"></path>
        <path class="dishline" d="M 66 18 L 66 31"></path>
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

  function buildStationBoard(container) {
    if (!container) return;
    container.innerHTML = "";
    for (let i = 1; i <= 12; i++) {
      const tile = document.createElement("div");
      tile.className = "stationtile";
      tile.id = `${container.id}-st-${i}`;
      tile.innerHTML = `
        <div class="stationhead">
          <div class="stationname">STATION ${String(i).padStart(2,"0")}</div>
          <div class="stationmeta" id="${container.id}-meta-${i}">—</div>
        </div>
        <div class="dishicon">${makeDishSVG()}</div>
        <div class="lightning">${makeLightningSVG()}</div>
      `;
      // station accent (for border glow etc.)
      tile.style.setProperty("--st-accent", ST_COLORS[i-1]);
      container.appendChild(tile);
    }
  }

  function updateStationBoard(container, visibleIds, pulsingIds) {
    if (!container) return;
    const visSet = new Set(visibleIds);
    const pulseSet = new Set(pulsingIds);
    for (let i = 1; i <= 12; i++) {
      const tile = $(`#${container.id}-st-${i}`);
      if (!tile) continue;
      const meta = $(`#${container.id}-meta-${i}`);
      const vis = visSet.has(i);
      const pul = pulseSet.has(i);
      tile.classList.toggle("is-visible", vis);
      tile.classList.toggle("is-pulsing", pul);
      if (meta) meta.textContent = vis ? "VISIBLE" : "—";
      // tint lightning per station
      const lightning = tile.querySelector(".lightning svg");
      if (lightning) lightning.style.fill = ST_COLORS[i-1];
    }
  }

  // -----------------------------
  // Matrices from UI
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
  // Running / speed
  // -----------------------------
  function setRunning(run) {
    app.running = !!run && !app.crashed;
    safeText(el.statusRun, app.running ? "RUNNING" : "PAUSED");
    if (el.btnToggleRun) el.btnToggleRun.textContent = app.running ? "Pause" : "Run";
    if (el.statusRun) {
      el.statusRun.classList.toggle("pill--warn", app.crashed);
      el.statusRun.classList.toggle("pill--idle", !app.crashed);
    }
  }

  function setSpeed(s) {
    app.speed = s;
    (el.spdChips || []).forEach(btn => {
      btn.classList.toggle("is-active", parseFloat(btn.dataset.speed) === s);
    });
  }

  // -----------------------------
  // Buffer helpers
  // -----------------------------
  function pushLimited(arr, val, max = app.bufMax) {
    arr.push(val);
    if (arr.length > max) arr.shift();
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
    pushLimited(app.neesHist, app.ekf.computeNEES(xt));
    pushLimited(app.nisHist, app.ekf.computeNIS());
    pushLimited(app.nisDfHist, app.ekf.lastInnov ? app.ekf.lastInnov.length : NaN);

    app.plotDirty = true;
  }

  // -----------------------------
  // Noise sampling
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
  // Readouts (continuous time)
  // -----------------------------
  function updateReadouts(visibleIds, tDisplay) {
    const tShown = Number.isFinite(tDisplay) ? tDisplay : app.t;

    safeText(el.readoutTime, `${fmt(tShown, 1)} s`);
    safeText(el.readoutStep, `${app.k}`);
    safeText(el.readoutVisible, `${visibleIds.length}`);

    // show the most recent measurement (if any)
    if (app.measT.length > 0) {
      const i = app.measT.length - 1;
      safeText(el.roRho,  `${fmt(app.measRho[i], 3)} km`);
      safeText(el.roRhod, `${fmt(app.measRhod[i], 5)} km/s`);
      safeText(el.roPhi,  `${fmt(app.measPhi[i], 5)} rad`);
    } else {
      safeText(el.roRho, "—");
      safeText(el.roRhod, "—");
      safeText(el.roPhi, "—");
    }

    safeText(el.roNEES, Number.isFinite(app.ekf.lastNEES) ? fmt(app.ekf.lastNEES, 2) : "—");
    safeText(el.roNIS,  Number.isFinite(app.ekf.lastNIS)  ? fmt(app.ekf.lastNIS, 2)  : "—");

    const dx = app.ekf.x[0] - app.truth[0];
    const dy = app.ekf.x[2] - app.truth[2];
    const dvx = app.ekf.x[1] - app.truth[1];
    const dvy = app.ekf.x[3] - app.truth[3];
    const nrm = Math.sqrt(dx*dx + dy*dy + dvx*dvx + dvy*dvy);
    safeText(el.roErrNorm, fmt(nrm, 4));
  }

  // -----------------------------
  // Signal visuals (pulse + squiggle)
  // -----------------------------
  function spawnSignalVisuals(xTrue, tNow, stationIds) {
    const now = performance.now();
    for (const id of stationIds) {
      const s = stationState(id - 1, tNow, app.Re, app.omega);
      const color = ST_COLORS[id-1];

      // dot-on-line pulse
      app.pulses.push({
        t0: now,
        dur: 380,
        sx: xTrue[0], sy: xTrue[2],
        tx: s.X, ty: s.Y,
        id, color
      });

      // squiggle line (data link)
      app.squiggles.push({
        t0: now,
        dur: 600,
        sx: xTrue[0], sy: xTrue[2],
        tx: s.X, ty: s.Y,
        id, color,
        phase: Math.random() * 10
      });
    }
    const cap = 260;
    if (app.pulses.length > cap) app.pulses.splice(0, app.pulses.length - cap);
    if (app.squiggles.length > cap) app.squiggles.splice(0, app.squiggles.length - cap);
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
    const k = 2; // ~2σ
    const rx = (k*s1) / app.cam.kmPerPx;
    const ry = (k*s2) / app.cam.kmPerPx;

    const c = app.cam.worldToScreen(xhat[0], xhat[2], W, H);

    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(angle);
    ctx.strokeStyle = "rgba(37,99,235,0.35)";
    ctx.fillStyle = "rgba(37,99,235,0.10)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(0.1, rx), Math.max(0.1, ry), 0, 0, 2*Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawSquiggle(ctx, x0, y0, x1, y1, amp, waves, phase) {
    const dx = x1 - x0, dy = y1 - y0;
    const L = Math.hypot(dx, dy) + 1e-9;
    const ux = dx / L, uy = dy / L;
    const nx = -uy, ny = ux;

    const steps = 42;
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const a = i / steps;
      const px = x0 + a*dx;
      const py = y0 + a*dy;
      const w = Math.sin(phase + a * waves * 2*Math.PI);
      const ox = amp * w * nx;
      const oy = amp * w * ny;
      if (i === 0) ctx.moveTo(px + ox, py + oy);
      else ctx.lineTo(px + ox, py + oy);
    }
    ctx.stroke();
  }

  function drawOrbitView() {
    const canvas = el.orbitCanvas;
    const ctx = app.orbitCtx;
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0,0,W,H);

    // Earth
    const earth = app.cam.worldToScreen(0,0,W,H);
    const RePx = Math.max(0.1, app.Re / app.cam.kmPerPx);
    ctx.save();
    ctx.beginPath();
    ctx.arc(earth.x, earth.y, RePx, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(249,115,22,0.08)";
    ctx.fill();
    ctx.strokeStyle = "rgba(249,115,22,0.22)";
    ctx.lineWidth = 1.3;
    ctx.stroke();
    ctx.restore();

    // Stations
    for (let i = 1; i <= 12; i++) {
      const s = stationState(i - 1, app.t, app.Re, app.omega);
      const p = app.cam.worldToScreen(s.X, s.Y, W, H);
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.0, 0, 2*Math.PI);
      ctx.fillStyle = ST_COLORS[i-1];
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.restore();
    }

    // Trajectories: truth & estimate
    const n = app.tHist.length;
    if (n > 2) {
      ctx.save();
      ctx.strokeStyle = "rgba(15,23,42,0.75)"; // truth: slate-900-ish
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
      ctx.strokeStyle = "rgba(37,99,235,0.85)"; // estimate: blue-600
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6,4]);
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
      if (started) ctx.stroke();
      ctx.restore();
    }

    // Markers: truth (white-ish) and estimate (blue)
    const satT = app.cam.worldToScreen(app.truth[0], app.truth[2], W, H);
    const satE = app.cam.worldToScreen(app.ekf.x[0], app.ekf.x[2], W, H);

    // truth marker
    ctx.save();
    ctx.beginPath();
    ctx.arc(satT.x, satT.y, 4.8, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(15,23,42,0.35)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();

    // estimate marker
    ctx.save();
    ctx.beginPath();
    ctx.arc(satE.x, satE.y, 4.6, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(37,99,235,0.90)";
    ctx.fill();
    ctx.strokeStyle = "rgba(37,99,235,0.30)";
    ctx.lineWidth = 1.0;
    ctx.stroke();
    ctx.restore();

    // Covariance ellipse around estimate
    drawCovEllipse(ctx, app.ekf.P, app.ekf.x, W, H);

    // Signals: pulses + squiggles
    const now = performance.now();

    // squiggles behind
    const squigStill = [];
    for (const s of app.squiggles) {
      const age = now - s.t0;
      if (age > s.dur) continue;
      squigStill.push(s);
      const a = age / s.dur;
      const w0 = app.cam.worldToScreen(s.sx, s.sy, W, H);
      const w1 = app.cam.worldToScreen(s.tx, s.ty, W, H);

      ctx.save();
      ctx.strokeStyle = s.color;
      ctx.globalAlpha = 0.22 * (1 - 0.35*a);
      ctx.lineWidth = 2.0;
      drawSquiggle(ctx, w0.x, w0.y, w1.x, w1.y, 4.0, 6, s.phase + a*10);
      ctx.restore();
    }
    app.squiggles = squigStill;

    // pulses on top
    const pulseStill = [];
    for (const p of app.pulses) {
      const age = now - p.t0;
      if (age > p.dur) continue;
      pulseStill.push(p);

      const a = age / p.dur;
      const s0 = app.cam.worldToScreen(p.sx, p.sy, W, H);
      const s1 = app.cam.worldToScreen(p.tx, p.ty, W, H);
      const x = s0.x + a * (s1.x - s0.x);
      const y = s0.y + a * (s1.y - s0.y);

      ctx.save();
      ctx.strokeStyle = p.color;
      ctx.globalAlpha = 0.18 * (1-a);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.stroke();

      ctx.fillStyle = p.color;
      ctx.globalAlpha = 0.85 * (1-a) + 0.15;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, 2*Math.PI);
      ctx.fill();
      ctx.restore();
    }
    app.pulses = pulseStill;

    // HUD
    ctx.save();
    ctx.fillStyle = "rgba(15,23,42,0.9)";
    ctx.font = "900 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`t = ${fmt(app.t,1)} s`, 10, 18);
    ctx.fillStyle = "rgba(71,85,105,0.95)";
    ctx.font = "800 11px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`zoom = ${fmt(app.cam.kmPerPx, 3)} km/px`, 10, 34);
    ctx.restore();
  }

  // -----------------------------
  // Plot rendering
  // -----------------------------
  function drawPlotsIfNeeded() {
    const now = performance.now();
    if (!app.plotDirty && (now - app.lastPlotDraw) < 200) return;
    app.lastPlotDraw = now;
    app.plotDirty = false;

    drawStatesPlot();
    drawErrorsPlot();
    drawMeasPlot();
    drawTrajPlot();
    drawConsPlot();
  }

  function plotBoxesForTile(x, y, w, h) {
    // Reserve left margin for y ticks and bottom for x ticks (inside drawAxes too)
    return { x, y, w, h };
  }

  function drawStatesPlot() {
    const ctx = app.ctxStates;
    const canvas = el.plotStates;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = (W - pad*3) / 2;
    const tileH = (H - pad*3) / 2;

    const labels = ["X", "Ẋ", "Y", "Ẏ"];
    const units  = ["km", "km/s", "km", "km/s"];

    const t = app.tHist;
    if (!t.length) return;
    const tMin = t[0], tMax = t[t.length - 1];

    for (let i = 0; i < 4; i++) {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = pad + col * (tileW + pad);
      const y = pad + row * (tileH + pad);

      const truth = app.xTrueHist[i];
      const est = app.xHatHist[i];
      const sig = app.sigHist[i];

      const up = new Array(est.length);
      const lo = new Array(est.length);
      for (let k = 0; k < est.length; k++) { up[k] = est[k] + 2*sig[k]; lo[k] = est[k] - 2*sig[k]; }

      const { mn, mx } = computeMinMax([truth, est, up, lo]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: `${labels[i]} (${units[i]})`,
        xLabel: "t (s)",
        yLabel: units[i],
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "", fmt: (v)=>fmt(v, (units[i]==="km")?0:3) }
      });

      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);
      plotLine(ctx, box, t, truth, tMin, tMax, mn, mx, { stroke: "rgba(15,23,42,0.75)", width: 1.5 });
      plotLine(ctx, box, t, est,   tMin, tMax, mn, mx, { stroke: "rgba(37,99,235,0.85)", width: 1.5 });
      plotLine(ctx, box, t, up,    tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, box, t, lo,    tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
    }
  }

  function drawErrorsPlot() {
    const ctx = app.ctxErrors;
    const canvas = el.plotErrors;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*5) / 4;

    const labels = ["X error (km)", "Ẋ error (km/s)", "Y error (km)", "Ẏ error (km/s)"];

    const t = app.tHist;
    if (!t.length) return;
    const tMin = t[0], tMax = t[t.length - 1];

    for (let i = 0; i < 4; i++) {
      const x = pad;
      const y = pad + i*(tileH + pad);

      const err = app.errHist[i];
      const sig = app.sigHist[i];

      const up = new Array(err.length);
      const lo = new Array(err.length);
      for (let k = 0; k < err.length; k++) { up[k] = 2*sig[k]; lo[k] = -2*sig[k]; }

      const { mn, mx } = computeMinMax([err, up, lo]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: labels[i],
        xLabel: (i===3) ? "t (s)" : "",
        yLabel: "",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "", fmt: (v)=>fmt(v,3) }
      });

      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);
      plotLine(ctx, box, t, err, tMin, tMax, mn, mx, { stroke: "rgba(37,99,235,0.85)", width: 1.5 });
      plotLine(ctx, box, t, up,  tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, box, t, lo,  tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });

      // zero line
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.x, box.y, box.w, box.h);
      ctx.clip();
      ctx.strokeStyle = "rgba(100,116,139,0.35)";
      ctx.lineWidth = 1;
      const y0 = box.y + box.h * (1 - (0 - mn) / (mx - mn));
      ctx.beginPath();
      ctx.moveTo(box.x, y0);
      ctx.lineTo(box.x + box.w, y0);
      ctx.stroke();
      ctx.restore();
    }
  }

  function computeTimeRangeForMeas() {
    // Use whichever series has data; prevents "right-half-only" when buffers shift.
    const candidates = [];
    if (app.tHist.length) { candidates.push(app.tHist[0], app.tHist[app.tHist.length-1]); }
    if (app.measT.length) { candidates.push(app.measT[0], app.measT[app.measT.length-1]); }
    if (app.predT.length) { candidates.push(app.predT[0], app.predT[app.predT.length-1]); }

    if (candidates.length < 2) return { tMin: 0, tMax: 1 };
    let tMin = Infinity, tMax = -Infinity;
    for (const v of candidates) {
      if (!Number.isFinite(v)) continue;
      tMin = Math.min(tMin, v);
      tMax = Math.max(tMax, v);
    }
    if (!Number.isFinite(tMin) || !Number.isFinite(tMax) || tMax <= tMin) return { tMin: 0, tMax: 1 };
    return { tMin, tMax };
  }

  function colorsFromStationIds(stIds, alpha = 0.85) {
    const out = new Array(stIds.length);
    for (let i = 0; i < stIds.length; i++) {
      const id = stIds[i];
      const c = (id>=1 && id<=12) ? ST_COLORS[id-1] : "#2563eb";
      // convert hex to rgba string
      const r = parseInt(c.slice(1,3),16);
      const g = parseInt(c.slice(3,5),16);
      const b = parseInt(c.slice(5,7),16);
      out[i] = `rgba(${r},${g},${b},${alpha})`;
    }
    return out;
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

    const { tMin, tMax } = computeTimeRangeForMeas();

    // Range
    {
      const x = pad, y = pad;
      const { mn, mx } = computeMinMax([app.measRho, app.predRho]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: "Range ρ (km) — measured vs predicted",
        xLabel: "",
        yLabel: "km",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "km", fmt: (v)=>fmt(v,0) }
      });
      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);

      const measColors = colorsFromStationIds(app.measStId, 0.85);
      plotScatter(ctx, box, app.measT, app.measRho, tMin, tMax, mn, mx, { r: 2.3 }, measColors);

      // predicted: slightly transparent black
      plotScatter(ctx, box, app.predT, app.predRho, tMin, tMax, mn, mx, { fill: "rgba(15,23,42,0.35)", r: 2.0 });
    }

    // Range-rate
    {
      const x = pad, y = pad + (tileH + pad);
      const { mn, mx } = computeMinMax([app.measRhod, app.predRhod]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: "Range-rate ρ̇ (km/s) — measured vs predicted",
        xLabel: "",
        yLabel: "km/s",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "km/s", fmt: (v)=>fmt(v,4) }
      });
      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);

      const measColors = colorsFromStationIds(app.measStId, 0.85);
      plotScatter(ctx, box, app.measT, app.measRhod, tMin, tMax, mn, mx, { r: 2.3 }, measColors);
      plotScatter(ctx, box, app.predT, app.predRhod, tMin, tMax, mn, mx, { fill: "rgba(15,23,42,0.35)", r: 2.0 });
    }

    // Angle
    {
      const x = pad, y = pad + 2*(tileH + pad);
      const { mn, mx } = computeMinMax([app.measPhi, app.predPhi]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: "Angle φ (rad) — measured vs predicted",
        xLabel: "t (s)",
        yLabel: "rad",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "rad", fmt: (v)=>fmt(v,3) }
      });
      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);

      const measColors = colorsFromStationIds(app.measStId, 0.85);
      plotScatter(ctx, box, app.measT, app.measPhi, tMin, tMax, mn, mx, { r: 2.3 }, measColors);
      plotScatter(ctx, box, app.predT, app.predPhi, tMin, tMax, mn, mx, { fill: "rgba(15,23,42,0.35)", r: 2.0 });
    }
  }

  function drawTrajPlot() {
    const ctx = app.ctxTraj;
    const canvas = el.plotTraj;
    if (!ctx || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const x0 = pad, y0 = pad;
    const w = W - pad*2, h = H - pad*2;

    // build extents from truth and estimate (XY)
    const Xs = app.xTrueHist[0].concat(app.xHatHist[0].filter(Number.isFinite));
    const Ys = app.xTrueHist[2].concat(app.xHatHist[2].filter(Number.isFinite));
    const { mn: xmin0, mx: xmax0 } = computeMinMax([Xs]);
    const { mn: ymin0, mx: ymax0 } = computeMinMax([Ys]);

    let xmin = xmin0, xmax = xmax0, ymin = ymin0, ymax = ymax0;
    const xr = xmax - xmin, yr = ymax - ymin;
    const r = Math.max(1e-6, Math.max(xr, yr));
    const xc = 0.5*(xmin + xmax);
    const yc = 0.5*(ymin + ymax);
    xmin = xc - 0.5*r; xmax = xc + 0.5*r;
    ymin = yc - 0.5*r; ymax = yc + 0.5*r;

    const ax = drawAxes(ctx, x0, y0, w, h, {
      title: "Inertial XY trajectory (km) — stations + links",
      xLabel: "X (km)",
      yLabel: "Y (km)",
      xTicks: { min: xmin, max: xmax, unit: "km", fmt: (v)=>fmt(v,0) },
      yTicks: { min: ymin, max: ymax, unit: "km", fmt: (v)=>fmt(v,0) }
    });

    const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);
    const sx = (X) => box.x + (X - xmin)/(xmax - xmin) * box.w;
    const sy = (Y) => box.y + box.h - (Y - ymin)/(ymax - ymin) * box.h;

    // clip
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();

    // Earth (guard against negative radius)
    const RePx = Math.max(0, (app.Re/(xmax-xmin))*(box.w));
    if (Number.isFinite(RePx) && RePx > 0) {
      ctx.beginPath();
      ctx.arc(sx(0), sy(0), RePx, 0, 2*Math.PI);
      ctx.fillStyle = "rgba(249,115,22,0.08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(249,115,22,0.22)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // Stations
    for (let i = 1; i <= 12; i++) {
      const s = stationState(i - 1, app.t, app.Re, app.omega);
      ctx.beginPath();
      ctx.arc(sx(s.X), sy(s.Y), 2.6, 0, 2*Math.PI);
      ctx.fillStyle = ST_COLORS[i-1];
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Truth path
    ctx.strokeStyle = "rgba(15,23,42,0.70)";
    ctx.lineWidth = 1.7;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < app.tHist.length; i++) {
      const X = app.xTrueHist[0][i], Y = app.xTrueHist[2][i];
      if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
      if (i === 0) ctx.moveTo(sx(X), sy(Y));
      else ctx.lineTo(sx(X), sy(Y));
    }
    ctx.stroke();

    // Estimate path
    ctx.strokeStyle = "rgba(37,99,235,0.85)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6,4]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < app.tHist.length; i++) {
      const X = app.xHatHist[0][i], Y = app.xHatHist[2][i];
      if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
      if (!started) { ctx.moveTo(sx(X), sy(Y)); started = true; }
      else ctx.lineTo(sx(X), sy(Y));
    }
    if (started) ctx.stroke();
    ctx.setLineDash([]);

    // If we have a current measurement set, draw links to visible stations
    const vis = visibleStations(app.truth, app.t, app.Re, app.omega);
    for (const id of vis) {
      const s = stationState(id - 1, app.t, app.Re, app.omega);
      ctx.strokeStyle = ST_COLORS[id-1];
      ctx.globalAlpha = 0.18;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(app.truth[0]), sy(app.truth[2]));
      ctx.lineTo(sx(s.X), sy(s.Y));
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Markers (truth + estimate)
    ctx.beginPath();
    ctx.arc(sx(app.truth[0]), sy(app.truth[2]), 4.2, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(15,23,42,0.35)";
    ctx.lineWidth = 1.1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(sx(app.ekf.x[0]), sy(app.ekf.x[2]), 4.0, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(37,99,235,0.90)";
    ctx.fill();
    ctx.restore();
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
    const t = app.tHist;
    const tMin = t.length ? t[0] : 0;
    const tMax = t.length ? t[t.length-1] : 1;

    // NEES df=4
    {
      const x = pad, y = pad;
      const df = 4;
      const lo = chi2Inv(alpha/2, df);
      const hi = chi2Inv(1 - alpha/2, df);
      const yLo = new Array(t.length).fill(lo);
      const yHi = new Array(t.length).fill(hi);

      const { mn, mx } = computeMinMax([app.neesHist, yLo, yHi]);
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: "NEES (df = 4) with χ² bounds",
        xLabel: "",
        yLabel: "",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "", fmt: (v)=>fmt(v,1) }
      });
      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);
      plotLine(ctx, box, t, app.neesHist, tMin, tMax, mn, mx, { stroke: "rgba(37,99,235,0.85)", width: 1.5 });
      plotLine(ctx, box, t, yLo, tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, box, t, yHi, tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
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
      const ax = drawAxes(ctx, x, y, tileW, tileH, {
        title: "NIS with χ² bounds (df = 3×visible)",
        xLabel: "t (s)",
        yLabel: "",
        xTicks: { min: tMin, max: tMax, unit: "s", fmt: (v)=>fmt(v,0) },
        yTicks: { min: mn, max: mx, unit: "", fmt: (v)=>fmt(v,1) }
      });
      const box = plotBoxesForTile(ax.plotX0, ax.plotY0, ax.plotW, ax.plotH);
      plotLine(ctx, box, t, app.nisHist, tMin, tMax, mn, mx, { stroke: "rgba(37,99,235,0.85)", width: 1.5 });
      plotLine(ctx, box, t, loArr, tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, box, t, hiArr, tMin, tMax, mn, mx, { stroke: "rgba(234,88,12,0.75)", width: 1.2, dash: [6,4] });
    }
  }

  // -----------------------------
  // Tabs
  // -----------------------------
  function setupTabs() {
    const panes = {
      states: el.paneStates,
      errors: el.paneErrors,
      meas: el.paneMeas,
      traj: el.paneTraj,
      cons: el.paneCons,
    };

    function activate(name) {
      (el.tabs || []).forEach(t => t.classList.toggle("is-active", t.dataset.tab === name));
      Object.entries(panes).forEach(([k, p]) => { if (p) p.classList.toggle("is-active", k === name); });

      // canvases change size when display toggles; rebind HiDPI and redraw
      handleResize(true);
    }

    (el.tabs || []).forEach(btn => {
      btn.addEventListener("click", () => activate(btn.dataset.tab));
    });

    // default active tab if none set
    const anyActive = (el.tabs || []).some(t => t.classList.contains("is-active"));
    if (!anyActive) activate("states");
  }

  // -----------------------------
  // Orbit canvas pan/zoom
  // -----------------------------
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
        const newKmPerPx = clamp(pinch.kmPerPx0 / scale, 2, 600);

        const mid = pinch.mid0;
        const before = app.cam.screenToWorld(mid.x, mid.y, W, H);
        app.cam.kmPerPx = newKmPerPx;
        const after = app.cam.screenToWorld(mid.x, mid.y, W, H);
        app.cam.cx += (before.x - after.x);
        app.cam.cy += (before.y - after.y);

        app.plotDirty = true;
        e.preventDefault();
        return;
      }

      if (!panning) return;

      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };

      app.cam.cx -= dx * app.cam.kmPerPx;
      app.cam.cy -= dy * app.cam.kmPerPx;

      app.plotDirty = true;
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
      app.cam.kmPerPx = clamp(app.cam.kmPerPx / zoom, 2, 600);
      const after = app.cam.screenToWorld(mx, my, W, H);
      app.cam.cx += (before.x - after.x);
      app.cam.cy += (before.y - after.y);

      app.plotDirty = true;
      e.preventDefault();
    }, { passive: false });

    if (el.btnResetView) el.btnResetView.addEventListener("click", () => app.cam.reset(app.Re));
    if (el.btnCenterView) el.btnCenterView.addEventListener("click", () => { app.cam.cx = 0; app.cam.cy = 0; });
    if (el.btnZoomCraft) el.btnZoomCraft.addEventListener("click", () => zoomToCraft());
  }

  function zoomToCraft() {
    // Put the truth spacecraft near center and zoom tighter.
    app.cam.cx = app.truth[0];
    app.cam.cy = app.truth[2];

    // zoom based on altitude, but clamp so it always looks good
    const r = Math.sqrt(app.truth[0]*app.truth[0] + app.truth[2]*app.truth[2]);
    const alt = Math.max(50, r - app.Re);
    // show ~2–3 Earth radii worth of local region around craft: smaller km/px = more zoom in
    const target = clamp(alt / 25, 2.5, 80);
    app.cam.kmPerPx = target;
  }

  // -----------------------------
  // Thrust controls
  // -----------------------------
  function readThrustMagnitude() {
    if (el.inpThrust) {
      const v = parseFloat(el.inpThrust.value);
      if (Number.isFinite(v) && v >= 0) return v;
    }
    return app.thrustMag;
  }

  function applyThrustFromInputs() {
    const mag = readThrustMagnitude();
    app.thrustMag = mag;

    const up = app.thrustKeys.up || app.thrustButtonsHeld.up;
    const down = app.thrustKeys.down || app.thrustButtonsHeld.down;
    const left = app.thrustKeys.left || app.thrustButtonsHeld.left;
    const right = app.thrustKeys.right || app.thrustButtonsHeld.right;

    let ax = 0, ay = 0;
    if (left) ax -= mag;
    if (right) ax += mag;
    if (up) ay += mag;
    if (down) ay -= mag;

    // normalize diagonals (keeps accel magnitude consistent)
    const n = Math.hypot(ax, ay);
    if (n > mag && n > 0) { ax = ax / n * mag; ay = ay / n * mag; }

    app.u[0] = ax;
    app.u[1] = ay;
  }

  function setupThrustControls() {
    // Keyboard: arrows or WASD
    window.addEventListener("keydown", (e) => {
      const k = e.key.toLowerCase();
      if (k === "arrowup" || k === "w") { app.thrustKeys.up = true; e.preventDefault(); }
      if (k === "arrowdown" || k === "s") { app.thrustKeys.down = true; e.preventDefault(); }
      if (k === "arrowleft" || k === "a") { app.thrustKeys.left = true; e.preventDefault(); }
      if (k === "arrowright" || k === "d") { app.thrustKeys.right = true; e.preventDefault(); }
      if (k === " " || k === "x") { // quick cut-off
        app.thrustKeys = { up:false, down:false, left:false, right:false };
        app.thrustButtonsHeld = { up:false, down:false, left:false, right:false };
        app.u[0] = 0; app.u[1] = 0;
      }
    }, { passive: false });

    window.addEventListener("keyup", (e) => {
      const k = e.key.toLowerCase();
      if (k === "arrowup" || k === "w") { app.thrustKeys.up = false; }
      if (k === "arrowdown" || k === "s") { app.thrustKeys.down = false; }
      if (k === "arrowleft" || k === "a") { app.thrustKeys.left = false; }
      if (k === "arrowright" || k === "d") { app.thrustKeys.right = false; }
    });

    // Button hold helpers
    function hold(btn, dir) {
      if (!btn) return;
      btn.addEventListener("pointerdown", (e) => {
        btn.setPointerCapture(e.pointerId);
        app.thrustButtonsHeld[dir] = true;
        e.preventDefault();
      }, { passive: false });
      btn.addEventListener("pointerup", () => { app.thrustButtonsHeld[dir] = false; });
      btn.addEventListener("pointercancel", () => { app.thrustButtonsHeld[dir] = false; });
      btn.addEventListener("pointerleave", () => { app.thrustButtonsHeld[dir] = false; });
    }
    hold(el.btnThUp, "up");
    hold(el.btnThDown, "down");
    hold(el.btnThLeft, "left");
    hold(el.btnThRight, "right");

    if (el.btnThZero) {
      el.btnThZero.addEventListener("click", () => {
        app.thrustKeys = { up:false, down:false, left:false, right:false };
        app.thrustButtonsHeld = { up:false, down:false, left:false, right:false };
        app.u[0] = 0; app.u[1] = 0;
      });
    }
  }

  // -----------------------------
  // Simulation step
  // -----------------------------
  function checkCrash(xTrueNext) {
    const r = Math.sqrt(xTrueNext[0]*xTrueNext[0] + xTrueNext[2]*xTrueNext[2]);
    return r <= (app.Re + 1.0); // 1 km guard band
  }

  function simulateOneStep() {
    if (app.crashed) return;

    // dt adjustable
    if (el.inpDt) {
      const dtUI = parseFloat(el.inpDt.value);
      if (Number.isFinite(dtUI)) app.dt = clamp(dtUI, 0.5, 120);
    }
    if (el.inpSteps) {
      const kUI = parseInt(el.inpSteps.value, 10);
      if (Number.isFinite(kUI)) app.Kmax = clamp(kUI, 1000, 10_000_000);
    }

    if (app.k >= app.Kmax) {
      setRunning(false);
      return;
    }

    // toggles
    app.enableProcNoise = !!el.tglProcNoise?.checked;
    app.enableMeasNoise = !!el.tglMeasNoise?.checked;

    // apply thrust state (from keys/buttons)
    applyThrustFromInputs();

    // EKF noise assumptions (live)
    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx?.value) || 1e-10, parseFloat(el.inpQkfAy?.value) || 1e-10),
      buildRstn(parseFloat(el.inpRkfRho?.value) || 0.05, parseFloat(el.inpRkfRhod?.value) || 4e-4, parseFloat(el.inpRkfPhi?.value) || 2e-3)
    );

    // truth noise (live)
    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx?.value) || 1e-10, parseFloat(el.inpQtrueAy?.value) || 1e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho?.value) || 0.05, parseFloat(el.inpRtrueRhod?.value) || 4e-4, parseFloat(el.inpRtruePhi?.value) || 2e-3);

    const dt = app.dt;
    const tNext = app.t + dt;

    // propagate truth
    let xTrueNext = rk4Step(app.truth, app.u, dt, app.mu);

    if (app.enableProcNoise) {
      const QdTrue = QdFromQacc(app.QtrueAcc, dt);
      const w = sampleMVN(4, QdTrue, app.randn);
      for (let i = 0; i < 4; i++) xTrueNext[i] += w[i];
    }

    // crash check
    if (checkCrash(xTrueNext)) {
      app.crashed = true;
      setRunning(false);
      safeText(el.statusRun, "STOPPED");
      // cut thrust
      app.u[0] = 0; app.u[1] = 0;
      return;
    }

    // visible stations
    const vis = visibleStations(xTrueNext, tNext, app.Re, app.omega);

    // build measurement vector
    let yMeas = new Float64Array(0);
    if (vis.length > 0) {
      yMeas = new Float64Array(3 * vis.length);

      for (let k = 0; k < vis.length; k++) {
        const m = measureOneStation(xTrueNext, tNext, vis[k], app.Re, app.omega);
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

        // wrap angle noise more realistically: it’s just additive in rad
        for (let i = 0; i < mDim; i++) yMeas[i] += v[i];
      }

      // store measurements per station (color mapping)
      for (let k = 0; k < vis.length; k++) {
        const stId = vis[k];
        pushLimited(app.measT, tNext);
        pushLimited(app.measRho, yMeas[3*k+0]);
        pushLimited(app.measRhod, yMeas[3*k+1]);
        pushLimited(app.measPhi, yMeas[3*k+2]);
        pushLimited(app.measStId, stId);
      }
    }

    // EKF update
    app.ekf.step(dt, tNext, app.u, yMeas, vis);

    // store predicted yhat (color it by station too)
    if (app.ekf.lastYhat && app.ekf.lastYhat.length > 0 && app.ekf.lastStationIds && app.ekf.lastStationIds.length) {
      const yhat = app.ekf.lastYhat;
      const st = app.ekf.lastStationIds;
      const n = Math.min(st.length, yhat.length / 3);

      for (let k = 0; k < n; k++) {
        pushLimited(app.predT, tNext);
        pushLimited(app.predRho, yhat[3*k+0]);
        pushLimited(app.predRhod, yhat[3*k+1]);
        pushLimited(app.predPhi, yhat[3*k+2]);
        pushLimited(app.predStId, st[k]);
      }
    }

    // visuals: measurement link pulses
    if (vis.length > 0) spawnSignalVisuals(xTrueNext, tNext, vis);

    // advance state
    app.truth = xTrueNext;
    app.t = tNext;
    app.k += 1;

    pushHistorySample();
    updateReadouts(vis, app.t);
    updateStationBoard(el.stationBoard, vis, vis);
    updateStationBoard(el.stationBoardSmall, vis, vis);
  }

  // -----------------------------
  // Reset / initialization
  // -----------------------------
  function hardResetAll() {
    app.crashed = false;

    // dt adjustable
    if (el.inpDt) {
      const dtUI = parseFloat(el.inpDt.value);
      if (Number.isFinite(dtUI)) app.dt = clamp(dtUI, 0.5, 120);
    } else {
      app.dt = 10;
    }

    if (el.inpSteps) {
      const kUI = parseInt(el.inpSteps.value, 10);
      if (Number.isFinite(kUI)) app.Kmax = clamp(kUI, 1000, 10_000_000);
      else app.Kmax = 1_000_000;
    } else {
      app.Kmax = 1_000_000;
    }

    app.enableProcNoise = !!el.tglProcNoise?.checked;
    app.enableMeasNoise = !!el.tglMeasNoise?.checked;

    // random seed every load/reset (as requested)
    const seed = (Math.floor(Math.random() * 1e9) >>> 0);
    app.rng = mulberry32(seed);
    app.randn = makeNormal(app.rng);

    // reset thrust
    app.u[0] = 0; app.u[1] = 0;
    app.thrustKeys = { up:false, down:false, left:false, right:false };
    app.thrustButtonsHeld = { up:false, down:false, left:false, right:false };
    if (el.inpThrust) {
      const v = parseFloat(el.inpThrust.value);
      app.thrustMag = Number.isFinite(v) ? v : app.thrustMag;
    }

    // noise defaults (the earlier version looked too “perfect”)
    // These defaults are intentionally larger; users can dial them down.
    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx?.value) || 5e-10, parseFloat(el.inpQtrueAy?.value) || 5e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho?.value) || 0.05, parseFloat(el.inpRtrueRhod?.value) || 4e-4, parseFloat(el.inpRtruePhi?.value) || 2e-3);

    // filter initial covariance
    const sigX0 = parseFloat(el.inpSigX0?.value) || 1e-3;
    const sigV0 = parseFloat(el.inpSigV0?.value) || 1e-4;
    const P0 = buildP0(sigX0, sigV0);

    // initial orbit: simple circular-ish LEO
    const alt = 500; // km
    const r0 = app.Re + alt;
    const vCirc = Math.sqrt(app.mu / r0);
    const x0True = new Float64Array([r0, 0, 0, vCirc]);

    // initial filter estimate offset
    const x0Hat = new Float64Array([x0True[0] + sigX0, x0True[1] + sigV0, x0True[2] - sigX0, x0True[3] - sigV0]);

    app.ekf = new EKF(app.mu, app.Re, app.omega);
    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx?.value) || 1e-10, parseFloat(el.inpQkfAy?.value) || 1e-10),
      buildRstn(parseFloat(el.inpRkfRho?.value) || 0.05, parseFloat(el.inpRkfRhod?.value) || 4e-4, parseFloat(el.inpRkfPhi?.value) || 2e-3)
    );
    app.ekf.reset(x0Hat, P0);

    app.t = 0;
    app.k = 0;
    app.truth = new Float64Array(x0True);

    // clear histories
    app.tHist = [];
    app.xTrueHist = [[],[],[],[]];
    app.xHatHist = [[],[],[],[]];
    app.sigHist = [[],[],[],[]];
    app.errHist = [[],[],[],[]];
    app.neesHist = [];
    app.nisHist = [];
    app.nisDfHist = [];

    app.measT = [];
    app.measRho = [];
    app.measRhod = [];
    app.measPhi = [];
    app.measStId = [];

    app.predT = [];
    app.predRho = [];
    app.predRhod = [];
    app.predPhi = [];
    app.predStId = [];

    app.pulses = [];
    app.squiggles = [];
    app.plotDirty = true;

    app.cam.reset(app.Re);
    setRunning(false);

    pushHistorySample();
    updateReadouts([], 0);
    updateStationBoard(el.stationBoard, [], []);
    updateStationBoard(el.stationBoardSmall, [], []);
  }

  function resetEKFOnly() {
    const sigX0 = parseFloat(el.inpSigX0?.value) || 1e-3;
    const sigV0 = parseFloat(el.inpSigV0?.value) || 1e-4;
    const P0 = buildP0(sigX0, sigV0);

    const xTrue = app.truth;
    const x0Hat = new Float64Array([xTrue[0] + sigX0, xTrue[1] + sigV0, xTrue[2] - sigX0, xTrue[3] - sigV0]);

    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx?.value) || 1e-10, parseFloat(el.inpQkfAy?.value) || 1e-10),
      buildRstn(parseFloat(el.inpRkfRho?.value) || 0.05, parseFloat(el.inpRkfRhod?.value) || 4e-4, parseFloat(el.inpRkfPhi?.value) || 2e-3)
    );
    app.ekf.reset(x0Hat, P0);

    // keep truth history but re-align EKF histories with NaNs
    const L = app.tHist.length;
    app.xHatHist = [[], [], [], []];
    app.sigHist = [[], [], [], []];
    app.errHist = [[], [], [], []];
    app.neesHist = [];
    app.nisHist = [];
    app.nisDfHist = [];

    for (let i = 0; i < L; i++) {
      for (let j = 0; j < 4; j++) {
        app.xHatHist[j].push(NaN);
        app.sigHist[j].push(NaN);
        app.errHist[j].push(NaN);
      }
      app.neesHist.push(NaN);
      app.nisHist.push(NaN);
      app.nisDfHist.push(NaN);
    }

    app.predT = [];
    app.predRho = [];
    app.predRhod = [];
    app.predPhi = [];
    app.predStId = [];

    app.plotDirty = true;
  }

  // -----------------------------
  // UI wiring
  // -----------------------------
  function setupSpeedButtons() {
    (el.spdChips || []).forEach(btn => {
      btn.addEventListener("click", () => setSpeed(parseFloat(btn.dataset.speed)));
    });
  }

  function setupControlButtons() {
    if (el.btnToggleRun) el.btnToggleRun.addEventListener("click", () => setRunning(!app.running));
    if (el.btnStep) el.btnStep.addEventListener("click", () => { if (!app.running) simulateOneStep(); });
    if (el.btnReset) el.btnReset.addEventListener("click", hardResetAll);
    if (el.btnApplyIC) el.btnApplyIC.addEventListener("click", hardResetAll);
    if (el.btnResetFilter) el.btnResetFilter.addEventListener("click", resetEKFOnly);
  }

  // -----------------------------
  // Resize handling
  // -----------------------------
  function handleResize(force = false) {
    app.orbitCtx = setupCanvasHiDPI(el.orbitCanvas);

    app.ctxStates = setupCanvasHiDPI(el.plotStates);
    app.ctxErrors = setupCanvasHiDPI(el.plotErrors);
    app.ctxMeas = setupCanvasHiDPI(el.plotMeas);
    app.ctxTraj = setupCanvasHiDPI(el.plotTraj);
    app.ctxCons = setupCanvasHiDPI(el.plotCons);

    app.plotDirty = true;
    if (force) {
      drawOrbitView();
      drawPlotsIfNeeded();
    }
  }

  // -----------------------------
  // Animation loop with real-time speed mapping
  // -----------------------------
  let lastRAF = 0;
  let simTimeAccumulator = 0; // seconds of simulated time to process

  function tickRAF(ts) {
    if (!lastRAF) lastRAF = ts;
    const dtReal = (ts - lastRAF) / 1000;
    lastRAF = ts;

    // continuous time display (even between discrete steps)
    const tContinuous = app.running ? (app.t + simTimeAccumulator) : app.t;
    const visNow = visibleStations(app.truth, app.t, app.Re, app.omega);
    updateReadouts(visNow, tContinuous);

    if (app.running) {
      simTimeAccumulator += dtReal * app.speed;

      const maxStepsPerFrame = 60;
      let steps = 0;
      while (simTimeAccumulator >= app.dt && steps < maxStepsPerFrame) {
        simulateOneStep();
        simTimeAccumulator -= app.dt;
        steps++;
      }
      if (steps === maxStepsPerFrame) {
        // avoid spiral-of-death; keep UI responsive
        simTimeAccumulator = 0;
      }
    } else {
      simTimeAccumulator = 0;
    }

    drawOrbitView();
    drawPlotsIfNeeded();
    requestAnimationFrame(tickRAF);
  }

  // -----------------------------
  // Bootstrap
  // -----------------------------
  function init() {
    // station boards (main + optional sidebar)
    buildStationBoard(el.stationBoard);
    buildStationBoard(el.stationBoardSmall);

    setupTabs();
    setupSpeedButtons();
    setupControlButtons();
    setupOrbitPanZoom();
    setupThrustControls();

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
