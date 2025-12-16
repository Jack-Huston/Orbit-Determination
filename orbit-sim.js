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
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i][j] = 0.5 * (A[i][j] + A[j][i]);
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
    // 12 stations equally spaced (theta0 = i*pi/6), i=0..11
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

    return { rho, rhod, phi, dX, dY, dXd, dYd };
  }

  function H_jacobian_forStations(x, t, stationIds, Re, omega) {
    // Builds stacked yhat (3*n) and H (3*n x 4)
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
      const HP = matMul(H, Pminus);                 // (m x 4)
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

    ctx.fillStyle = "rgba(223,244,255,0.9)";
    ctx.font = "700 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillText(title, 10, 16);

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
  // App state + DOM handles
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
    speed: 10,           // default: fast demo
    dt: 10,
    Kmax: 1400,
    mu: 398600,
    Re: 6378,
    omega: 2*Math.PI/86400,
    seed: 100,

    u: new Float64Array([0,0]),

    t: 0,
    k: 0,
    truth: new Float64Array(4),
    ekf: new EKF(398600, 6378, 2*Math.PI/86400),

    enableProcNoise: true,
    enableMeasNoise: true,
    QtrueAcc: eye(2),
    RtrueStn: eye(3),

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

    pulses: [], // {t0,dur,sx,sy,tx,ty,id}

    cam: new Camera(),
    orbitCtx: null,

    ctxStates: null,
    ctxErrors: null,
    ctxMeas: null,
    ctxTraj: null,
    ctxCons: null,

    plotDirty: true,
    lastPlotDraw: 0,

    rng: null,
    randn: null,
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
  function updateStationBoard(visibleIds, pulsingIds) {
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
  // Build matrices from UI
  // -----------------------------
  function nominalState(altKm, velFactor, mu, Re) {
    const r0 = Re + altKm;
    const vCirc = r0 * Math.sqrt(mu / Math.pow(r0,3));
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

  // -----------------------------
  // Running / speed
  // -----------------------------
  function setRunning(run) {
    app.running = run;
    el.statusRun.textContent = run ? "RUNNING" : "PAUSED";
    el.btnToggleRun.textContent = run ? "Pause" : "Run";
    el.statusRun.style.borderColor = run ? "rgba(120,243,182,.35)" : "rgba(110,231,255,.22)";
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
  // Buffer helpers
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
  // Noise sampling
  // -----------------------------
  function sampleMVN(dim, cov, randn) {
    // Cholesky for small symmetric PSD
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
    el.readoutTime.textContent = `${fmt(app.t, 1)} s`;
    el.readoutStep.textContent = `${app.k}`;
    el.readoutVisible.textContent = `${visibleIds.length}`;

    if (app.measT.length > 0) {
      const i = app.measT.length - 1;
      el.roRho.textContent = `${fmt(app.measRho[i], 3)} km`;
      el.roRhod.textContent = `${fmt(app.measRhod[i], 5)} km/s`;
      el.roPhi.textContent = `${fmt(app.measPhi[i], 5)} rad`;
    } else {
      el.roRho.textContent = "—";
      el.roRhod.textContent = "—";
      el.roPhi.textContent = "—";
    }

    el.roNEES.textContent = Number.isFinite(app.ekf.lastNEES) ? fmt(app.ekf.lastNEES, 2) : "—";
    el.roNIS.textContent = Number.isFinite(app.ekf.lastNIS) ? fmt(app.ekf.lastNIS, 2) : "—";

    const dx = app.ekf.x[0] - app.truth[0];
    const dy = app.ekf.x[2] - app.truth[2];
    const dvx = app.ekf.x[1] - app.truth[1];
    const dvy = app.ekf.x[3] - app.truth[3];
    const nrm = Math.sqrt(dx*dx + dy*dy + dvx*dvx + dvy*dvy);
    el.roErrNorm.textContent = fmt(nrm, 4);
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
    ctx.ellipse(0, 0, rx, ry, 0, 0, 2*Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawOrbitView() {
    const canvas = el.orbitCanvas;
    const ctx = app.orbitCtx;
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0,0,W,H);

    // Earth
    const earth = app.cam.worldToScreen(0,0,W,H);
    const RePx = app.Re / app.cam.kmPerPx;
    ctx.save();
    ctx.beginPath();
    ctx.arc(earth.x, earth.y, RePx, 0, 2*Math.PI);
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
      ctx.arc(p.x, p.y, 3.0, 0, 2*Math.PI);
      ctx.fillStyle = "rgba(120,243,182,0.85)";
      ctx.fill();
      ctx.restore();
    }

    // Trajectories
    const n = app.tHist.length;
    if (n > 2) {
      // truth
      ctx.save();
      ctx.strokeStyle = "rgba(231,238,252,0.90)";
      ctx.lineWidth = 1.5;
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

      // estimate
      ctx.save();
      ctx.strokeStyle = "rgba(110,231,255,0.90)";
      ctx.lineWidth = 1.5;
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

    // Current markers
    const satT = app.cam.worldToScreen(app.truth[0], app.truth[2], W, H);
    const satE = app.cam.worldToScreen(app.ekf.x[0], app.ekf.x[2], W, H);

    ctx.save();
    ctx.beginPath();
    ctx.arc(satT.x, satT.y, 4.5, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(231,238,252,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(231,238,252,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(satE.x, satE.y, 4.5, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(110,231,255,0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(110,231,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Cov ellipse
    drawCovEllipse(ctx, app.ekf.P, app.ekf.x, W, H);

    // Signal pulses
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
      ctx.strokeStyle = `rgba(255,204,102,${0.20 * (1-a)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.stroke();

      ctx.fillStyle = `rgba(255,204,102,${0.85 * (1-a) + 0.15})`;
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, 2*Math.PI);
      ctx.fill();
      ctx.restore();
    }
    app.pulses = still;

    // HUD
    ctx.save();
    ctx.fillStyle = "rgba(223,244,255,0.92)";
    ctx.font = "700 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`t = ${fmt(app.t,1)} s`, 10, 18);
    ctx.fillStyle = "rgba(127,138,161,0.95)";
    ctx.font = "600 11px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`km/px = ${fmt(app.cam.kmPerPx, 3)}`, 10, 34);
    ctx.restore();
  }

  // -----------------------------
  // Plot rendering
  // -----------------------------
  function drawPlotsIfNeeded() {
    const now = performance.now();
    if (!app.plotDirty && (now - app.lastPlotDraw) < 250) return;
    app.lastPlotDraw = now;
    app.plotDirty = false;

    drawStatesPlot();
    drawErrorsPlot();
    drawMeasPlot();
    drawTrajPlot();
    drawConsPlot();
  }

  function drawStatesPlot() {
    const ctx = app.ctxStates;
    if (!ctx) return;
    const canvas = el.plotStates;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = (W - pad*3) / 2;
    const tileH = (H - pad*3) / 2;

    const labels = ["X (km)", "Xdot (km/s)", "Y (km)", "Ydot (km/s)"];
    for (let i = 0; i < 4; i++) {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = pad + col * (tileW + pad);
      const y = pad + row * (tileH + pad);

      drawAxes(ctx, x, y, tileW, tileH, { title: labels[i], xLabel: "t (s)" });

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
      const tMin = t[0], tMax = t[t.length - 1];

      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, truth, tMin, tMax, mn, mx, { stroke: "rgba(231,238,252,0.92)", width: 1.5 });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, est,   tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.5 });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, up,    tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.70)", width: 1.2, dash: [6,4] });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, lo,    tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.70)", width: 1.2, dash: [6,4] });
    }
  }

  function drawErrorsPlot() {
    const ctx = app.ctxErrors;
    if (!ctx) return;
    const canvas = el.plotErrors;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*5) / 4;

    const labels = ["X error (km)", "Xdot error (km/s)", "Y error (km)", "Ydot error (km/s)"];
    for (let i = 0; i < 4; i++) {
      const x = pad;
      const y = pad + i*(tileH + pad);

      drawAxes(ctx, x, y, tileW, tileH, { title: labels[i], xLabel: i===3 ? "t (s)" : "" });

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
      const tMin = t[0], tMax = t[t.length - 1];

      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, err, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.5 });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, up,  tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, lo,  tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });

      // zero line
      ctx.save();
      ctx.strokeStyle = "rgba(110,231,255,0.10)";
      ctx.lineWidth = 1;
      const y0 = y + 26 + (tileH-34) * (1 - (0 - mn) / (mx - mn));
      ctx.beginPath();
      ctx.moveTo(x+8, y0);
      ctx.lineTo(x+tileW-8, y0);
      ctx.stroke();
      ctx.restore();
    }
  }

  function drawMeasPlot() {
    const ctx = app.ctxMeas;
    if (!ctx) return;
    const canvas = el.plotMeas;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*5) / 3;

    const tMin = app.tHist.length ? app.tHist[0] : 0;
    const tMax = app.tHist.length ? app.tHist[app.tHist.length-1] : 1;

    // Range
    {
      const x = pad, y = pad;
      drawAxes(ctx, x, y, tileW, tileH, { title: "Range ρ (km) — measured (blue) vs predicted (red)", xLabel: "" });

      const { mn, mx } = computeMinMax([app.measRho, app.predRho]);
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.measT, app.measRho, tMin, tMax, mn, mx, { fill: "rgba(58,166,255,0.90)", r: 2.2 });
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.predT, app.predRho, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.85)", r: 2.2 });
    }

    // Range-rate
    {
      const x = pad, y = pad + (tileH + pad);
      drawAxes(ctx, x, y, tileW, tileH, { title: "Range-rate ρ̇ (km/s) — measured (blue) vs predicted (red)", xLabel: "" });

      const { mn, mx } = computeMinMax([app.measRhod, app.predRhod]);
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.measT, app.measRhod, tMin, tMax, mn, mx, { fill: "rgba(58,166,255,0.90)", r: 2.2 });
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.predT, app.predRhod, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.85)", r: 2.2 });
    }

    // Angle
    {
      const x = pad, y = pad + 2*(tileH + pad);
      drawAxes(ctx, x, y, tileW, tileH, { title: "Angle φ (rad) — measured (blue) vs predicted (red)", xLabel: "t (s)" });

      const { mn, mx } = computeMinMax([app.measPhi, app.predPhi]);
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.measT, app.measPhi, tMin, tMax, mn, mx, { fill: "rgba(58,166,255,0.90)", r: 2.2 });
      plotScatter(ctx, x+8, y+26, tileW-16, tileH-34, app.predT, app.predPhi, tMin, tMax, mn, mx, { fill: "rgba(255,107,107,0.85)", r: 2.2 });
    }
  }

  function drawTrajPlot() {
    const ctx = app.ctxTraj;
    if (!ctx) return;
    const canvas = el.plotTraj;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const x0 = pad, y0 = pad;
    const w = W - pad*2, h = H - pad*2;
    drawAxes(ctx, x0, y0, w, h, { title: "XY Trajectory (km) — Truth (white) vs EKF (cyan) + Earth + Stations", xLabel: "X (km)", yLabel: "Y (km)" });

    const Xs = app.xTrueHist[0].concat(app.xHatHist[0].filter(Number.isFinite));
    const Ys = app.xTrueHist[2].concat(app.xHatHist[2].filter(Number.isFinite));
    const { mn: xmin0, mx: xmax0 } = computeMinMax([Xs]);
    const { mn: ymin0, mx: ymax0 } = computeMinMax([Ys]);

    let xmin = xmin0, xmax = xmax0, ymin = ymin0, ymax = ymax0;
    const xr = xmax - xmin, yr = ymax - ymin;
    const r = Math.max(xr, yr);
    const xc = 0.5*(xmin + xmax);
    const yc = 0.5*(ymin + ymax);
    xmin = xc - 0.5*r;
    xmax = xc + 0.5*r;
    ymin = yc - 0.5*r;
    ymax = yc + 0.5*r;

    const sx = (X) => x0 + 8 + (X - xmin)/(xmax - xmin) * (w - 16);
    const sy = (Y) => y0 + 26 + (h - 34) - (Y - ymin)/(ymax - ymin) * (h - 34);

    // Earth
    ctx.save();
    const RePx = (app.Re/(xmax-xmin))*(w-16);
    ctx.beginPath();
    ctx.arc(sx(0), sy(0), RePx, 0, 2*Math.PI);
    ctx.fillStyle = "rgba(120,243,182,0.10)";
    ctx.fill();
    ctx.strokeStyle = "rgba(120,243,182,0.25)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();

    // Stations
    for (let i = 1; i <= 12; i++) {
      const s = stationState(i - 1, app.t, app.Re, app.omega);
      ctx.save();
      ctx.beginPath();
      ctx.arc(sx(s.X), sy(s.Y), 2.2, 0, 2*Math.PI);
      ctx.fillStyle = "rgba(120,243,182,0.85)";
      ctx.fill();
      ctx.restore();
    }

    // Truth path
    ctx.save();
    ctx.strokeStyle = "rgba(231,238,252,0.92)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < app.tHist.length; i++) {
      const X = app.xTrueHist[0][i], Y = app.xTrueHist[2][i];
      if (i === 0) ctx.moveTo(sx(X), sy(Y));
      else ctx.lineTo(sx(X), sy(Y));
    }
    ctx.stroke();
    ctx.restore();

    // Estimate path
    ctx.save();
    ctx.strokeStyle = "rgba(110,231,255,0.92)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < app.tHist.length; i++) {
      const X = app.xHatHist[0][i], Y = app.xHatHist[2][i];
      if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
      if (!started) { ctx.moveTo(sx(X), sy(Y)); started = true; }
      else ctx.lineTo(sx(X), sy(Y));
    }
    ctx.stroke();
    ctx.restore();

    // Current markers
    ctx.save();
    ctx.fillStyle = "rgba(231,238,252,0.95)";
    ctx.beginPath();
    ctx.arc(sx(app.truth[0]), sy(app.truth[2]), 3.5, 0, 2*Math.PI);
    ctx.fill();
    ctx.fillStyle = "rgba(110,231,255,0.95)";
    ctx.beginPath();
    ctx.arc(sx(app.ekf.x[0]), sy(app.ekf.x[2]), 3.5, 0, 2*Math.PI);
    ctx.fill();
    ctx.restore();
  }

  function drawConsPlot() {
    const ctx = app.ctxCons;
    if (!ctx) return;
    const canvas = el.plotCons;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    clearPanel(ctx, W, H);

    const pad = 12;
    const tileW = W - pad*2;
    const tileH = (H - pad*4) / 2;

    const alpha = clamp(parseFloat(el.inpAlpha.value) || 0.05, 0.01, 0.2);
    const t = app.tHist;
    const tMin = t.length ? t[0] : 0;
    const tMax = t.length ? t[t.length-1] : 1;

    // NEES df=4
    {
      const x = pad, y = pad;
      drawAxes(ctx, x, y, tileW, tileH, { title: "NEES (df=4) with χ² bounds", xLabel: "" });

      const df = 4;
      const lo = chi2Inv(alpha/2, df);
      const hi = chi2Inv(1 - alpha/2, df);
      const yLo = new Array(t.length).fill(lo);
      const yHi = new Array(t.length).fill(hi);

      const { mn, mx } = computeMinMax([app.neesHist, yLo, yHi]);
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, app.neesHist, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.5 });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, yLo, tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, yHi, tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });
    }

    // NIS df=3*visible
    {
      const x = pad, y = pad + tileH + pad;
      drawAxes(ctx, x, y, tileW, tileH, { title: "NIS with χ² bounds (df = 3×visible)", xLabel: "t (s)" });

      const loArr = [];
      const hiArr = [];
      for (let i = 0; i < app.nisDfHist.length; i++) {
        const df = app.nisDfHist[i];
        if (!Number.isFinite(df) || df <= 0) { loArr.push(NaN); hiArr.push(NaN); continue; }
        loArr.push(chi2Inv(alpha/2, df));
        hiArr.push(chi2Inv(1 - alpha/2, df));
      }

      const { mn, mx } = computeMinMax([app.nisHist, loArr, hiArr]);
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, app.nisHist, tMin, tMax, mn, mx, { stroke: "rgba(110,231,255,0.92)", width: 1.5 });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, loArr, tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });
      plotLine(ctx, x+8, y+26, tileW-16, tileH-34, t, hiArr, tMin, tMax, mn, mx, { stroke: "rgba(255,204,102,0.75)", width: 1.2, dash: [6,4] });
    }
  }

  // -----------------------------
  // Tabs
  // -----------------------------
  function setupTabs() {
    const tabs = document.querySelectorAll(".tab");
    const panes = {
      states: $("#pane-states"),
      errors: $("#pane-errors"),
      meas: $("#pane-meas"),
      traj: $("#pane-traj"),
      cons: $("#pane-cons"),
    };

    function activate(name) {
      for (const t of tabs) t.classList.toggle("is-active", t.dataset.tab === name);
      for (const [k, p] of Object.entries(panes)) p.classList.toggle("is-active", k === name);
      // redraw because canvas sizes sometimes change after display toggles
      handleResize(true);
    }

    tabs.forEach(btn => {
      btn.addEventListener("click", () => activate(btn.dataset.tab));
    });
  }

  // -----------------------------
  // Draggable orbit card
  // -----------------------------
  function setupOrbitCardDrag() {
    const card = el.orbitCard;
    const handle = el.orbitDragHandle;

    // store initial if not already
    const init = { left: card.offsetLeft, top: card.offsetTop };
    card.dataset.initLeft = String(init.left);
    card.dataset.initTop = String(init.top);

    let dragging = false;
    let startX = 0, startY = 0;
    let startLeft = 0, startTop = 0;

    function onDown(e) {
      // avoid stealing events from buttons
      if (e.target && (e.target.closest(".iconbtn") || e.target.closest("button"))) return;
      dragging = true;
      card.classList.add("is-dragging");
      const p = getPointerXY(e);
      startX = p.x;
      startY = p.y;
      startLeft = card.offsetLeft;
      startTop = card.offsetTop;
      window.addEventListener("pointermove", onMove, { passive: false });
      window.addEventListener("pointerup", onUp, { passive: true });
      window.addEventListener("pointercancel", onUp, { passive: true });
      e.preventDefault();
    }

    function onMove(e) {
      if (!dragging) return;
      const p = getPointerXY(e);
      const dx = p.x - startX;
      const dy = p.y - startY;

      // constrain within viewport
      const maxLeft = Math.max(0, window.innerWidth - card.offsetWidth - 8);
      const maxTop = Math.max(0, window.innerHeight - card.offsetHeight - 8);
      const newLeft = clamp(startLeft + dx, 0, maxLeft);
      const newTop = clamp(startTop + dy, 0, maxTop);

      card.style.left = `${newLeft}px`;
      card.style.top = `${newTop}px`;
      card.style.position = "absolute";
      e.preventDefault();
    }

    function onUp() {
      dragging = false;
      card.classList.remove("is-dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    }

    handle.addEventListener("pointerdown", onDown, { passive: false });

    el.btnResetLayout.addEventListener("click", () => {
      card.style.position = "absolute";
      card.style.left = `${card.dataset.initLeft || 12}px`;
      card.style.top = `${card.dataset.initTop || 12}px`;
    });
  }

  function getPointerXY(e) {
    if (e.touches && e.touches.length) {
      return { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
    return { x: e.clientX, y: e.clientY };
  }

  // -----------------------------
  // Orbit canvas pan/zoom (mouse + touch pinch)
  // -----------------------------
  function setupOrbitPanZoom() {
    const canvas = el.orbitCanvas;

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
        // start pinch
        const ids = Array.from(pointers.keys());
        const A = pointers.get(ids[0]);
        const B = pointers.get(ids[1]);
        pinch.active = true;
        pinch.idA = ids[0];
        pinch.idB = ids[1];
        pinch.dist0 = Math.hypot(B.x - A.x, B.y - A.y);
        pinch.kmPerPx0 = app.cam.kmPerPx;

        const rect = canvas.getBoundingClientRect();
        const mid = { x: (A.x + B.x)/2 - rect.left, y: (A.y + B.y)/2 - rect.top };
        pinch.mid0 = mid;

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

        // zoom about pinch midpoint
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

    el.btnResetView.addEventListener("click", () => app.cam.reset(app.Re));
    el.btnCenterView.addEventListener("click", () => {
      app.cam.cx = 0;
      app.cam.cy = 0;
    });
  }

  // -----------------------------
  // Simulation step
  // -----------------------------
  function spawnSignalPulses(xTrue, tNow, stationIds) {
    for (const id of stationIds) {
      const s = stationState(id - 1, tNow, app.Re, app.omega);
      app.pulses.push({
        t0: performance.now(),
        dur: 320,
        sx: xTrue[0],
        sy: xTrue[2],
        tx: s.X,
        ty: s.Y,
        id
      });
    }
    if (app.pulses.length > 200) app.pulses.splice(0, app.pulses.length - 200);
  }

  function simulateOneStep() {
    if (app.k >= app.Kmax) {
      setRunning(false);
      return;
    }

    // live params
    app.dt = clamp(parseFloat(el.inpDt.value) || app.dt, 1, 120);
    app.Kmax = clamp(parseInt(el.inpSteps.value || String(app.Kmax), 10), 50, 20000);
    app.mu = parseFloat(el.inpMu.value) || app.mu;
    app.Re = parseFloat(el.inpRe.value) || app.Re;
    app.omega = parseFloat(el.inpOmega.value) || app.omega;

    app.enableProcNoise = !!el.tglProcNoise.checked;
    app.enableMeasNoise = !!el.tglMeasNoise.checked;
    app.u[0] = parseFloat(el.inpUx.value) || 0;
    app.u[1] = parseFloat(el.inpUy.value) || 0;

    // update EKF constants
    app.ekf.mu = app.mu;
    app.ekf.Re = app.Re;
    app.ekf.omega = app.omega;
    app.ekf.setQR(
      buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10),
      buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4)
    );

    // truth noise matrices
    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx.value) || 1e-10, parseFloat(el.inpQtrueAy.value) || 1e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho.value) || 0.01, parseFloat(el.inpRtrueRhod.value) || 1e-4, parseFloat(el.inpRtruePhi.value) || 5e-4);

    const dt = app.dt;
    const tNext = app.t + dt;

    // Truth propagation
    let xTrueNext = rk4Step(app.truth, app.u, dt, app.mu);

    if (app.enableProcNoise) {
      const QdTrue = QdFromQacc(app.QtrueAcc, dt);
      const w = sampleMVN(4, QdTrue, app.randn);
      for (let i = 0; i < 4; i++) xTrueNext[i] += w[i];
    }

    // Visible stations at tNext
    const vis = visibleStations(xTrueNext, tNext, app.Re, app.omega);

    // Measurements
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
        for (let i = 0; i < mDim; i++) yMeas[i] += v[i];
      }

      // store measurements scatter
      for (let k = 0; k < vis.length; k++) {
        pushLimited(app.measT, tNext);
        pushLimited(app.measRho, yMeas[3*k+0]);
        pushLimited(app.measRhod, yMeas[3*k+1]);
        pushLimited(app.measPhi, yMeas[3*k+2]);
      }
    }

    // EKF update
    app.ekf.step(dt, tNext, app.u, yMeas, vis);

    // Store predicted measurement scatter
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

    // pulses (measurement taken this step)
    if (vis.length > 0) spawnSignalPulses(xTrueNext, tNext, vis);

    // advance
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

    app.QtrueAcc = buildQacc(parseFloat(el.inpQtrueAx.value) || 1e-10, parseFloat(el.inpQtrueAy.value) || 1e-10);
    app.RtrueStn = buildRstn(parseFloat(el.inpRtrueRho.value) || 0.01, parseFloat(el.inpRtrueRhod.value) || 1e-4, parseFloat(el.inpRtruePhi.value) || 5e-4);

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
    const x0Hat = new Float64Array([xNom0[0] + sigX0, xNom0[1] + sigV0, xNom0[2] + sigX0, xNom0[3] + sigV0]);

    app.ekf = new EKF(app.mu, app.Re, app.omega);
    const QkfAcc = buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10);
    const RkfStn = buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4);
    app.ekf.setQR(QkfAcc, RkfStn);
    app.ekf.reset(x0Hat, P0);

    app.t = 0;
    app.k = 0;
    app.truth = new Float64Array(x0True);

    app.rng = mulberry32(app.seed);
    app.randn = makeNormal(app.rng);

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

    app.pulses = [];
    app.plotDirty = true;

    app.cam.reset(app.Re);

    setRunning(false);
    pushHistorySample();
    updateReadouts([]);
    updateStationBoard([], []);
  }

function resetEKFOnly() {
  const sigX0 = parseFloat(el.inpSigX0.value) || 1e-6;
  const sigV0 = parseFloat(el.inpSigV0.value) || 1e-5;
  const P0 = buildP0(sigX0, sigV0);
  const xTrue = app.truth;
  const x0Hat = new Float64Array([xTrue[0] + sigX0, xTrue[1] + sigV0, xTrue[2] + sigX0, xTrue[3] + sigV0]);

  app.ekf.setQR(
    buildQacc(parseFloat(el.inpQkfAx.value) || 1e-10, parseFloat(el.inpQkfAy.value) || 1e-10),
    buildRstn(parseFloat(el.inpRkfRho.value) || 0.01, parseFloat(el.inpRkfRhod.value) || 1e-4, parseFloat(el.inpRkfPhi.value) || 5e-4)
  );
  app.ekf.reset(x0Hat, P0);

  // wipe EKF-dependent histories (keep truth history)
  app.xHatHist = [[], [], [], []];
  app.sigHist = [[], [], [], []];
  app.errHist = [[], [], [], []];
  app.neesHist = [];
  app.nisHist = [];
  app.nisDfHist = [];
  app.predT = [];
  app.predRho = [];
  app.predRhod = [];
  app.predPhi = [];

  // align lengths with tHist using NaNs up to current point
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
// Meas/plot panes already implemented earlier in file:
// drawStatesPlot/drawErrorsPlot/drawMeasPlot/drawTrajPlot/drawConsPlot
// -----------------------------

// -----------------------------
// UI wiring
// -----------------------------
function setupSpeedButtons() {
  const group = document.querySelectorAll(".chip[data-speed]");
  group.forEach((btn) => {
    btn.addEventListener("click", () => setSpeed(parseFloat(btn.dataset.speed)));
  });
}

function setupControlButtons() {
  el.btnToggleRun.addEventListener("click", () => setRunning(!app.running));
  el.btnStep.addEventListener("click", () => {
    if (!app.running) simulateOneStep();
  });
  el.btnReset.addEventListener("click", hardResetAll);
  el.btnApplyIC.addEventListener("click", hardResetAll);
  el.btnResetFilter.addEventListener("click", resetEKFOnly);
}

// -----------------------------
// Resize handling (HiDPI canvas)
// -----------------------------
function handleResize(force = false) {
  // Orbit canvas
  app.orbitCtx = setupCanvasHiDPI(el.orbitCanvas);

  // Plot canvases
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
// Definition: speed = 1 means dt seconds of sim time takes dt seconds real time.
// So sim-time rate = speed * 1.0 seconds per second.
let lastRAF = 0;
let simTimeAccumulator = 0; // seconds of simulated time to process

function tickRAF(ts) {
  if (!lastRAF) lastRAF = ts;
  const dtReal = (ts - lastRAF) / 1000;
  lastRAF = ts;

  if (app.running) {
    // accumulate desired simulated time (sec)
    simTimeAccumulator += dtReal * app.speed;

    // step in fixed chunks of app.dt
    // guard against long frame: cap how many steps per frame
    const maxStepsPerFrame = 50;
    let steps = 0;
    while (simTimeAccumulator >= app.dt && steps < maxStepsPerFrame) {
      simulateOneStep();
      simTimeAccumulator -= app.dt;
      steps++;
    }
    if (steps === maxStepsPerFrame) {
      // drop leftover to prevent spiral of death
      simTimeAccumulator = 0;
    }
  } else {
    // when paused, don't accumulate (prevents jump on resume)
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
  buildStationBoard();
  setupTabs();
  setupSpeedButtons();
  setupControlButtons();
  setupOrbitCardDrag();
  setupOrbitPanZoom();

  // default: 10× highlighted
  setSpeed(10);

  handleResize(true);
  window.addEventListener("resize", () => handleResize(true));

  hardResetAll();
  requestAnimationFrame(tickRAF);
}

// Start once DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}
})();

