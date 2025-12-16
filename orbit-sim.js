/* orbit-sim.js
   Client-side orbit + EKF simulator (2D).
   No external libs, no network calls, no eval.
*/
(() => {
  "use strict";

  // -----------------------------
  // Utilities: DOM helpers
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
  // Small linear algebra (dense)
  // -----------------------------
  function zeros(r, c) {
    const A = new Array(r);
    for (let i = 0; i < r; i++) {
      const row = new Float64Array(c);
      A[i] = row;
    }
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
      // pivot
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
      // normalize
      const piv = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= piv;
      // eliminate
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
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) B[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
    return B;
  }

  // -----------------------------
  // Statistics: normal quantile & chi-square bounds (Wilson-Hilferty)
  // -----------------------------
  // Acklam inverse normal CDF approximation
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

  // Wilson-Hilferty approx for chi-square quantile
  function chi2Inv(p, k) {
    const z = normInv(p);
    const a = 2 / (9 * k);
    const q = k * Math.pow(1 - a + z * Math.sqrt(a), 3);
    return Math.max(0, q);
  }

  // -----------------------------
  // Dynamics + Measurement models
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
    // 12 stations equally spaced (theta0 = (i) * pi/6), 1-indexed in MATLAB
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
      if (dot > 0) vis.push(i+1); // return 1..12
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

    return { rho, rhod, phi, dX, dY, dXd, dYd, sX: s.X, sY: s.Y, sXd: s.Xd, sYd: s.Yd };
  }

  function H_jacobian_forStations(x, t, stationIds, Re, omega) {
    // Builds stacked yhat (3*n) and H (3*n x 4) using your MATLAB formulas.
    const n = stationIds.length;
    const yhat = new Float64Array(3*n);
    const H = zeros(3*n, 4);

    for (let k = 0; k < n; k++) {
      const id = stationIds[k];
      const m = measureOneStation(x, t, id, Re, omega);

      const rho = m.rho;
      const a = m.dX*m.dXd + m.dY*m.dYd;
      const rho3 = rho*rho*rho;

      // y
      yhat[3*k + 0] = m.rho;
      yhat[3*k + 1] = m.rhod;
      yhat[3*k + 2] = m.phi;

      // H_i
      // Range
      H[3*k + 0][0] = m.dX / rho;
      H[3*k + 0][2] = m.dY / rho;

      // Range-rate
      H[3*k + 1][0] = (m.dXd / rho) - (a * m.dX / rho3);
      H[3*k + 1][2] = (m.dYd / rho) - (a * m.dY / rho3);
      H[3*k + 1][1] = m.dX / rho;
      H[3*k + 1][3] = m.dY / rho;

      // Angle
      const rho2 = rho*rho;
      H[3*k + 2][0] = -m.dY / rho2;
      H[3*k + 2][2] =  m.dX / rho2;
    }

    return { yhat, H };
  }

  // Discrete-time process noise mapping for acceleration noise -> state
  // L = [[dt^2/2,0],[dt,0],[0,dt^2/2],[0,dt]]
  function QdFromQacc(Qacc2x2, dt) {
    const h = 0.5 * dt * dt;
    const L = zeros(4,2);
    L[0][0] = h;   L[1][0] = dt;
    L[2][1] = h;   L[3][1] = dt;

    // Qd = L Qacc L'
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

    step(dt, tNext, u, yMeas /* Float64Array stacked */, stationIds /* array */) {
      // Prediction: propagate nonlinear state
      const xminus = rk4Step(this.x, u, dt, this.mu);

      // Cov prediction with linearization: F = I + dt*A
      const A = A_jacobian(this.x, this.mu);
      const F = matAdd(eye(4), matScale(A, dt));

      const Qd = QdFromQacc(this.Qacc, dt);
      let Pminus = matAdd(matMul(matMul(F, this.P), matTrans(F)), Qd);
      Pminus = symmetrize(Pminus);

      // If no measurement, accept prediction
      if (!yMeas || yMeas.length === 0 || stationIds.length === 0) {
        this.x = xminus;
        this.P = Pminus;
        this.lastInnov = null;
        this.lastS = null;
        this.lastYhat = null;
        this.lastStationIds = [];
        return;
      }

      // Measurement prediction + Jacobian
      const { yhat, H } = H_jacobian_forStations(xminus, tNext, stationIds, this.Re, this.omega);

      // Build R block diag
      const n = stationIds.length;
      const m = 3*n;
      const R = zeros(m, m);
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
          R[3*k + i][3*k + j] = this.Rstn[i][j];
        }
      }

      // Innovation e = y - yhat (wrap angles)
      const e = new Float64Array(m);
      for (let i = 0; i < m; i++) e[i] = yMeas[i] - yhat[i];
      for (let k = 0; k < n; k++) {
        const angIdx = 3*k + 2;
        e[angIdx] = wrapToPi(e[angIdx]);
      }

      // S = H P H' + R
      const HP = matMul(H, Pminus);          // (m x 4)
      const S = matAdd(matMul(HP, matTrans(H)), R); // (m x m)
      const Ssym = symmetrize(S);

      // K = P H' S^-1
      const PHt = matMul(Pminus, matTrans(H)); // (4 x m)
      const Sinv = matInv(Ssym);
      const K = matMul(PHt, Sinv); // (4 x m)

      // xplus = xminus + K e
      const Ke = matMulVec(K, e);
      const xplus = new Float64Array(4);
      for (let i = 0; i < 4; i++) xplus[i] = xminus[i] + Ke[i];

      // Pplus = (I - K H) Pminus
      const KH = matMul(K, H);
      const I = eye(4);
      const IKH = matSub(I, KH);
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
      // e' Pinv e
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
  // Plotting (Canvas)
  // -----------------------------
  function setupCanvasHiDPI(canvas) {
    const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const w = Math.floor(rect.width * dpr);
    const h = Math.floor(rect.height * dpr);
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
    // subtle vignette
    const g = ctx.createRadialGradient(w*0.5, h*0.4, 10, w*0.5, h*0.5, Math.max(w,h)*0.8);
    g.addColorStop(0, "rgba(58,166,255,0.06)");
    g.addColorStop(1, "rgba(0,0,0,0.0)");
    ctx.fillStyle = g;
    ctx.fillRect(0,0,w,h);
  }

  function drawAxes(ctx, x, y, w, h, opts) {
    const { grid = true, title = "", xLabel = "", yLabel = "" } = opts || {};
    ctx.save();
    ctx.translate(x,y);

    // frame
    ctx.strokeStyle = "rgba(110,231,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0,0,w,h);

    if (grid) {
      ctx.strokeStyle = "rgba(110,231,255,0.06)";
      ctx.lineWidth = 1;
      const nx = 6, ny = 4;
      for (let i = 1; i < nx; i++) {
        const gx = (i/nx)*w;
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, h);
        ctx.stroke();
      }
      for (let j = 1; j < ny; j++) {
        const gy = (j/ny)*h;
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }
    }

    // title
    ctx.fillStyle = "rgba(223,244,255,0.9)";
    ctx.font = "700 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillText(title, 10, 16);

    // labels
    ctx.fillStyle = "rgba(127,138,161,0.9)";
    ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
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
    const sx = (tt) => x0 + (tt - tMin) / (tMax - tMin) * w;
    const sy = (vv) => y0 + h - (vv - yMin) / (yMax - yMin) * h;

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
    const sx = (tt) => x0 + (tt - tMin) / (tMax - tMin) * w;
    const sy = (vv) => y0 + h - (vv - yMin) / (yMax - yMin) * h;

    const r = style.r || 2.2;
    ctx.save();
    ctx.fillStyle = style.fill || "rgba(58,166,255,0.95)";
    for (let i = 0; i < n; i++) {
      const px = sx(t[i]);
      const py = sy(y[i]);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2*Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // -----------------------------
  // Orbit View: camera + interactions + rendering
  // -----------------------------
  class Camera {
    constructor() {
      this.cx = 0;
      this.cy = 0;
      this.kmPerPx = 35; // zoom (lower = zoom in)
    }
    reset(Re) {
      this.cx = 0;
      this.cy = 0;
      this.kmPerPx = Math.max(10, Re / 140); // reasonable default
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
  // Application state
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

    inpDt: $("#inpDt"),
    inpSteps: $("#inpSteps"),
    inpMu: $("#inpMu"),
    inpRe: $("#inpRe"),
    inpOmega: $("#inpOmega"),
    inpSeed: $("#inpSeed"),

    inpAlt: $("#inpAlt"),
    inpVelFactor: $("#inpVelFactor"),
    inpDx: $("#inpDx"),
    inpDy: $("#inpDy"),
    inpDvx: $("#inpDvx"),
    inpDvy: $("#inpDvy"),

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

    inpUx: $("#inpUx"),
    inpUy: $("#inpUy"),

    orbitCard: $("#orbitCard"),
    orbitDragHandle: $("#orbitDragHandle"),
    orbitCanvas: $("#orbitCanvas"),
    btnCenterView: $("#btnCenterView"),
    btnResetView: $("#btnResetView"),
    btnResetLayout: $("#btnResetLayout"),

    plotStates: $("#plotStates"),
    plotErrors: $("#plotErrors"),
    plotMeas: $("#plotMeas"),
    plotTraj: $("#plotTraj"),
    plotCons: $("#plotCons"),
  };

  const app = {
    running: false,
    speed: 10, // default for demo; 1x is real-time
    dt: 10,
    Kmax: 1400,
    mu: 398600,
    Re: 6378,
    omega: 2*Math.PI/86400,
    seed: 100,

    u: new Float64Array([0,0]),

    // truth + ekf
    t: 0,
    k: 0,
    truth: new Float64Array(4),
    ekf: new EKF(398600, 6378, 2*Math.PI/86400),

    // noise
    enableProcNoise: true,
    enableMeasNoise: true,
    QtrueAcc: eye(2),
    RtrueStn: eye(3),

    // buffers for plotting
    bufMax: 2200,
    tHist: [],
    xTrueHist: [[],[],[],[]],
    xHatHist: [[],[],[],[]],
    sigHist: [[],[],[],[]],
    errHist: [[],[],[],[]],
    neesHist: [],
    nisHist: [],
    nisDfHist: [],

    measT: [],
    measRho: [],
    measRhod: [],
    measPhi: [],
    predT: [],
    predRho: [],
    predRhod: [],
    predPhi: [],

    visibleHist: [],

    // pulses for orbit view
    pulses: [], // {t0, dur, x0,y0,x1,y1}
    lastVisible: [],

    // camera + interactions
    cam: new Camera(),
    orbitCtx: null,

    // plot ctx
    ctxStates: null,
    ctxErrors: null,
    ctxMeas: null,
    ctxTraj: null,
    ctxCons: null,

    plotDirty: true,
    lastPlotDraw: 0,
  };

  // -----------------------------
  // Station board UI
  // -----------------------------
  function makeDishSVG() {
    return `
      <svg viewBox="0 0 120 34" aria-hidden="true" focusable="false">
        <path class="dishfill" d="M 12 26 Q 60 6 108 26 Q 60 30 12 26 Z"></path>
        <path class="dishline" d="M 12 26 Q 60 6 108 26"></path>
        <path class="dishline" d="M 58 26 L 50 10"></path>
        <path class="dishline" d="M 60 26 L 60 9"></path>
        <path class="dishline" d="M 62 26 L 70 10"></path>
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

  function buildStationBoard() {
    el.stationBoard.innerHTML = "";
    for (let i = 1; i <= 12; i++) {
      const tile = document.createElement("div");
      tile.className = "stationtile";
      tile.id = `st-${i}`;
      tile.innerHTML = `
        <div class="stationhead">
          <div class="stationname">STATION ${String(i).padStart(2,"0")}</div>
          <div class="stationmeta" id="stmeta-${i}">—</div>
        </div>
        <div class="dishicon">${makeDishSVG()}</div>
        <div class="lightning">${makeLightningSVG()}</div>
      `;
      el.stationBoard.appendChild(tile);
    }
  }

  function updateStationBoard(visibleIds, pulsingIds, t) {
    const visSet = new Set(visibleIds);
    const pulseSet = new Set(pulsingIds);
    for (let i = 1; i <= 12; i++) {
      const tile = $(`#st-${i}`);
      if (!tile) continue;
      tile.classList.toggle("is-visible", visSet.has(i));
      tile.classList.toggle("is-pulsing", pulseSet.has(i));
      const meta = $(`#stmeta-${i}`);
      if (meta) {
        meta.textContent = visSet.has(i) ? `VISIBLE` : `—`;
      }
    }
  }

  // -----------------------------
  // Init/reset
  // -----------------------------
  function nominalState(altKm, velFactor, mu, Re) {
    const r0 = Re + altKm;
    const vCirc = r0 * Math.sqrt(mu / Math.pow(r0,3));
    // Place at (r0, 0), velocity in +Y direction => [X, Xdot, Y, Ydot] = [r0, 0, 0, v]
    return new Float64Array([r0, 0, 0, vCirc * velFactor]);
  }

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

  function hardResetAll() {
    // pull params
    app.dt = clamp(parseFloat(el.inpDt.value) || 10, 1, 120);
    app.Kmax = clamp(parseInt(el.inpSteps.value || "1400", 10), 50, 20000);
    app.mu = parseFloat(el.inpMu.value) || 398600;
    app.Re = parseFloat(el.inpRe.value) || 6378;
    app.omega = parseFloat(el.inpOmega.value) || (2*Math.PI/86400);
    app.seed = (parseInt(el.inpSeed.value || "100", 10) >>> 0);

    app.enableProcNoise = !!el.tglProcNoise.checked;
    app.enableMeasNoise = !!el.tglMeasNoise.checked;

    app.u[0] = parseFloat(el.inpUx.value) || 0;
    app.u[1] = parseFloat(el.inpUy.value) || 0;

    // truth noise
    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx.value) || 1e-10, parseFloat(el.inpQtrueAy.value) || 1e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho.value) || 0.01, parseFloat(el.inpRtrueRhod.value) || 1e-4, parseFloat(el.inpRtruePhi.value) || 5e-4);

    // EKF settings
    const sigX0 = parseFloat(el.inpSigX0.value) || 1e-6;
    const sigV0 = parseFloat(el.inpSigV0.value) || 1e-5;
    const P0 = buildP0(sigX0, sigV0);

    const alt = parseFloat(el.inpAlt.value) || 300;
    const velFactor = parseFloat(el.inpVelFactor.value) || 1.0;
    const dx = parseFloat(el.inpDx.value) || 0;
    const dy = parseFloat(el.inpDy.value) || 0;
    const dvx = parseFloat(el.inpDvx.value) || 0;
    const dvy = parseFloat(el.inpDvy.value) || 0;

    const xNom0 = nominalState(alt, velFactor, app.mu, app.Re);
    const x0True = new Float64Array([xNom0[0] + dx, xNom0[1] + dvx, xNom0[2] + dy, xNom0[3] + dvy]);

    // Start EKF slightly perturbed like your script
    const x0Hat = new Float64Array([xNom0[0] + sigX0, xNom0[1] + sigV0, xNom0[2] + sigX0, xNom0[3] + sigV0]);

    // update EKF constants
    app.ekf = new EKF(app.mu, app.Re, app.omega);
    const QkfAcc = buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10);
    const RkfStn = buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4);
    app.ekf.setQR(QkfAcc, RkfStn);
    app.ekf.reset(x0Hat, P0);

    // truth init
    app.t = 0;
    app.k = 0;
    app.truth = new Float64Array(x0True);

    // RNG
    app.rng = mulberry32(app.seed);
    app.randn = makeNormal(app.rng);

    // buffers reset
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
    app.predT = [];
    app.predRho = [];
    app.predRhod = [];
    app.predPhi = [];
    app.visibleHist = [];
    app.pulses = [];
    app.lastVisible = [];
    app.plotDirty = true;

    // camera
    app.cam.reset(app.Re);

    // status
    setRunning(false);
    pushHistorySample(); // initial sample
    updateReadouts([], []);
    updateStationBoard([], [], app.t);
  }

  function resetEKFOnly() {
    const sigX0 = parseFloat(el.inpSigX0.value) || 1e-6;
    const sigV0 = parseFloat(el.inpSigV0.value) || 1e-5;
    const P0 = buildP0(sigX0, sigV0);

    // set to nominal truth at current time + small offset
    const xTrue = app.truth;
    const x0Hat = new Float64Array([xTrue[0] + sigX0, xTrue[1] + sigV0, xTrue[2] + sigX0, xTrue[3] + sigV0]);

    const QkfAcc = buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10);
    const RkfStn = buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4);
    app.ekf.setQR(QkfAcc, RkfStn);
    app.ekf.reset(x0Hat, P0);

    // wipe EKF-related histories but keep truth timeline
    app.xHatHist = [[],[],[],[]];
    app.sigHist = [[],[],[],[]];
    app.errHist = [[],[],[],[]];
    app.neesHist = [];
    app.nisHist = [];
    app.nisDfHist = [];
    app.predT = [];
    app.predRho = [];
    app.predRhod = [];
    app.predPhi = [];

    // rebuild samples from current truth history length (keep tHist)
    for (let idx = 0; idx < app.tHist.length; idx++) {
      const t = app.tHist[idx];
      // placeholder: we won't reconstruct the old EKF; start fresh from now
      // Fill arrays with NaN to keep plotting coherent
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
  // Speed controls
  // -----------------------------
  function setSpeed(s) {
    app.speed = s;
    for (const id of ["spd1","spd10","spd100","spd1000"]) {
      const b = $(`#${id}`);
      if (!b) continue;
      b.classList.toggle("is-active", parseFloat(b.dataset.speed) === s);
    }
  }

  function setRunning(run) {
    app.running = run;
    el.statusRun.textContent = run ? "RUNNING" : "PAUSED";
    el.btnToggleRun.textContent = run ? "Pause" : "Run";
    el.statusRun.style.borderColor = run ? "rgba(120,243,182,.35)" : "rgba(110,231,255,.22)";
  }

  // -----------------------------
  // History buffer management
  // -----------------------------
  function pushLimited(arr, val) {
    arr.push(val);
    if (arr.length > app.bufMax) arr.shift();
  }
  function pushHistorySample() {
    const t = app.t;
    const xt = app.truth;
    const xh = app.ekf.x;

    pushLimited(app.tHist, t);
    for (let i = 0; i < 4; i++) {
      pushLimited(app.xTrueHist[i], xt[i]);
      pushLimited(app.xHatHist[i], xh[i]);
      // sigma from P
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
  // Simulation step
  // -----------------------------
  function sampleMVN(dim, cov, randn) {
    // Cholesky for small symmetric PSD (simple, with jitter)
    const n = dim;
    const A = matCopy(cov);
    for (let i = 0; i < n; i++) A[i][i] += 1e-18;

    // Cholesky lower
    const L = zeros(n,n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k]*L[j][k];
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(0, s));
        } else {
          L[i][j] = s / (L[j][j] + 1e-18);
        }
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

  function simulateOneStep() {
    if (app.k >= app.Kmax) {
      setRunning(false);
      return;
    }

    // update live params that can change while running
    app.dt = clamp(parseFloat(el.inpDt.value) || app.dt, 1, 120);
    app.mu = parseFloat(el.inpMu.value) || app.mu;
    app.Re = parseFloat(el.inpRe.value) || app.Re;
    app.omega = parseFloat(el.inpOmega.value) || app.omega;
    app.enableProcNoise = !!el.tglProcNoise.checked;
    app.enableMeasNoise = !!el.tglMeasNoise.checked;
    app.u[0] = parseFloat(el.inpUx.value) || 0;
    app.u[1] = parseFloat(el.inpUy.value) || 0;

    // Update EKF noise matrices live
    app.ekf.mu = app.mu;
    app.ekf.Re = app.Re;
    app.ekf.omega = app.omega;
    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10),
      buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4)
    );

    // Truth noise matrices live
    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx.value) || 1e-10, parseFloat(el.inpQtrueAy.value) || 1e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho.value) || 0.01, parseFloat(el.inpRtrueRhod.value) || 1e-4, parseFloat(el.inpRtruePhi.value) || 5e-4);

    const dt = app.dt;
    const tNext = app.t + dt;

    // Propagate truth
    let xTrueNext = rk4Step(app.truth, app.u, dt, app.mu);

    // Add process noise (truth)
    if (app.enableProcNoise) {
      const QdTrue = QdFromQacc(app.QtrueAcc, dt);
      const w = sampleMVN(4, QdTrue, app.randn);
      for (let i = 0; i < 4; i++) xTrueNext[i] += w[i];
    }

    // Determine visible stations at tNext (matches your data alignment)
    const vis = visibleStations(xTrueNext, tNext, app.Re, app.omega);

    // Measurements from truth
    let yMeas = new Float64Array(0);
    const pulsingIds = vis.slice(); // measurement attempt for every visible station each step

    if (vis.length > 0) {
      yMeas = new Float64Array(3 * vis.length);
      for (let k = 0; k < vis.length; k++) {
        const m = measureOneStation(xTrueNext, tNext, vis[k], app.Re, app.omega);
        yMeas[3*k+0] = m.rho;
        yMeas[3*k+1] = m.rhod;
        yMeas[3*k+2] = m.phi;
      }

      // add measurement noise
      if (app.enableMeasNoise) {
        // Build block R, sample noise
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

      // store measurement scatter arrays (per-station points)
      // For plotting like your MATLAB: scatter all station pass points vs time
      for (let k = 0; k < vis.length; k++) {
        pushLimited(app.measT, tNext);
        pushLimited(app.measRho, yMeas[3*k+0]);
        pushLimited(app.measRhod, yMeas[3*k+1]);
        pushLimited(app.measPhi, yMeas[3*k+2]);
      }
    }

    // EKF step with measurements at tNext
    app.ekf.step(dt, tNext, app.u, yMeas, vis);

    // Store predicted measurements (EKF yhat-) if available
    if (app.ekf.lastYhat && app.ekf.lastYhat.length > 0) {
      const yhat = app.ekf.lastYhat;
      const n = yhat.length / 3;
      for (let k = 0; k < n; k++) {
        pushLimited(app.predT, tNext);
        pushLimited(app.predRho, yhat[3*k+0]);
        pushLimited(app.predRhod, yhat[3*k+1]);
        pushLimited(app.predPhi, yhat[3*k+2]);
      }
    }

    // Add orbit-view signal pulses
    spawnSignalPulses(xTrueNext, tNext, pulsingIds);

    // Advance
    app.truth = xTrueNext;
    app.t = tNext;
    app.k += 1;

    pushHistorySample();
    updateReadouts(vis, pulsingIds);
    updateStationBoard(vis, pulsingIds, tNext);
  }

  function spawnSignalPulses(xTrue, tNow, stationIds) {
    // Convert world coords to canvas coords later at render time.
    // Store pulses in world coords.
    for (const id of stationIds) {
      const s = stationState(id - 1, tNow, app.Re, app.omega);
      app.pulses.push({
        t0: performance.now(),
        dur: 320, // ms
        sx: xTrue[0],
        sy: xTrue[2],
        tx: s.X,
        ty: s.Y,
        id
      });
    }
    // prune
    if (app.pulses.length > 200) app.pulses.splice(0, app.pulses.length - 200);
  }
