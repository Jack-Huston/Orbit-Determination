/* orbit-sim.js
   Client-side 2D orbit + EKF + linear controller demo.
   Lucide used only for station icon.
*/
(() => {
  'use strict';

  // ---------- Constants (fixed) ----------
  const MU = 398600.0;              // km^3/s^2
  const R_E = 6378.0;               // km
  const OMEGA_E = 7.2722052e-5;     // rad/s
  const G0 = 9.80665e-3;            // km/s^2

  const TWO_PI = Math.PI * 2;
  const EPS = 1e-12;

  // ---------- Small utils ----------
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  function wrapPi(a) {
    let x = a;
    x = (x + Math.PI) % (TWO_PI);
    if (x < 0) x += TWO_PI;
    return x - Math.PI;
  }

  function safeNum(v, fallback) {
    const x = Number(v);
    return Number.isFinite(x) ? x : fallback;
  }

  // Normal inverse CDF (Acklam approximation)
  function normInv(p) {
    const pp = clamp(p, 1e-12, 1 - 1e-12);
    const a = [
      -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    ];
    const b = [
      -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01
    ];
    const c = [
      -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    ];
    const d = [
      7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00
    ];

    const plow = 0.02425;
    const phigh = 1 - plow;
    let q, r;

    if (pp < plow) {
      q = Math.sqrt(-2 * Math.log(pp));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (pp > phigh) {
      q = Math.sqrt(-2 * Math.log(1 - pp));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }

    q = pp - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // Chi-square inverse CDF (Wilson–Hilferty transform)
  function chi2Inv(p, k) {
    const z = normInv(p);
    const kk = Math.max(1e-9, k);
    const a = 2 / (9 * kk);
    const term = 1 - a + z * Math.sqrt(a);
    return kk * term * term * term;
  }

  // Deterministic RNG for reproducibility
  function XorShift32(seed) {
    let x = seed >>> 0;
    const nextU32 = () => {
      x ^= (x << 13); x >>>= 0;
      x ^= (x >>> 17); x >>>= 0;
      x ^= (x << 5); x >>>= 0;
      return x >>> 0;
    };
    return {
      nextU32,
      nextFloat: () => nextU32() / 4294967296
    };
  }

  function randn(rng) {
    if (randn._hasSpare) {
      randn._hasSpare = false;
      return randn._spare;
    }
    let u = 0, v = 0;
    while (u < 1e-12) u = rng();
    while (v < 1e-12) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(TWO_PI * v);
    const z1 = mag * Math.sin(TWO_PI * v);
    randn._spare = z1;
    randn._hasSpare = true;
    return z0;
  }
  randn._hasSpare = false;
  randn._spare = 0;

  // ---------- Linear algebra (small, allocation-free) ----------
  function mat4Identity(out) {
    out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
  }

  function mat4Copy(out, A) {
    for (let i = 0; i < 16; i++) out[i] = A[i];
    return out;
  }

  function mat4Mul(out, A, B) {
    const a00 = A[0], a01 = A[1], a02 = A[2], a03 = A[3];
    const a10 = A[4], a11 = A[5], a12 = A[6], a13 = A[7];
    const a20 = A[8], a21 = A[9], a22 = A[10], a23 = A[11];
    const a30 = A[12], a31 = A[13], a32 = A[14], a33 = A[15];

    const b00 = B[0], b01 = B[1], b02 = B[2], b03 = B[3];
    const b10 = B[4], b11 = B[5], b12 = B[6], b13 = B[7];
    const b20 = B[8], b21 = B[9], b22 = B[10], b23 = B[11];
    const b30 = B[12], b31 = B[13], b32 = B[14], b33 = B[15];

    out[0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    out[1] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    out[2] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    out[3] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;

    out[4] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    out[5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    out[6] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    out[7] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;

    out[8] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    out[9] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    out[10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    out[11] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;

    out[12] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
    out[13] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
    out[14] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
    out[15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
    return out;
  }

  function mat4MulT(out, A, B) {
    // out = A * B^T
    const b00 = B[0], b10 = B[4], b20 = B[8], b30 = B[12];
    const b01 = B[1], b11 = B[5], b21 = B[9], b31 = B[13];
    const b02 = B[2], b12 = B[6], b22 = B[10], b32 = B[14];
    const b03 = B[3], b13 = B[7], b23 = B[11], b33 = B[15];

    for (let r = 0; r < 4; r++) {
      const a0 = A[4 * r + 0], a1 = A[4 * r + 1], a2 = A[4 * r + 2], a3 = A[4 * r + 3];
      out[4 * r + 0] = a0 * b00 + a1 * b01 + a2 * b02 + a3 * b03;
      out[4 * r + 1] = a0 * b10 + a1 * b11 + a2 * b12 + a3 * b13;
      out[4 * r + 2] = a0 * b20 + a1 * b21 + a2 * b22 + a3 * b23;
      out[4 * r + 3] = a0 * b30 + a1 * b31 + a2 * b32 + a3 * b33;
    }
    return out;
  }

  function mat4Symmetrize(P) {
    const p01 = 0.5 * (P[1] + P[4]);
    const p02 = 0.5 * (P[2] + P[8]);
    const p03 = 0.5 * (P[3] + P[12]);
    const p12 = 0.5 * (P[6] + P[9]);
    const p13 = 0.5 * (P[7] + P[13]);
    const p23 = 0.5 * (P[11] + P[14]);

    P[1] = p01; P[4] = p01;
    P[2] = p02; P[8] = p02;
    P[3] = p03; P[12] = p03;
    P[6] = p12; P[9] = p12;
    P[7] = p13; P[13] = p13;
    P[11] = p23; P[14] = p23;
  }

  function mat4AddDiag(P, d) {
    P[0] += d;
    P[5] += d;
    P[10] += d;
    P[15] += d;
  }

  function cholDecompInPlace(S, n) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = S[i * n + j];
        for (let k = 0; k < j; k++) {
          sum -= S[i * n + k] * S[j * n + k];
        }
        if (i === j) {
          if (sum <= 0 || !Number.isFinite(sum)) return false;
          S[i * n + i] = Math.sqrt(sum);
        } else {
          const lij = sum / S[j * n + j];
          if (!Number.isFinite(lij)) return false;
          S[i * n + j] = lij;
        }
      }
      for (let j = i + 1; j < n; j++) S[i * n + j] = 0;
    }
    return true;
  }

  function cholSolve(L, n, b, x) {
    // forward solve: L y = b  (reuse x as y)
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let k = 0; k < i; k++) sum -= L[i * n + k] * x[k];
      x[i] = sum / L[i * n + i];
    }
    // back solve: L^T x = y
    for (let i = n - 1; i >= 0; i--) {
      let sum = x[i];
      for (let k = i + 1; k < n; k++) sum -= L[k * n + i] * x[k];
      x[i] = sum / L[i * n + i];
    }
    return x;
  }

  function matInv4(P, out) {
    const A = new Float64Array(16);
    for (let i = 0; i < 16; i++) A[i] = P[i];
    mat4Identity(out);

    for (let c = 0; c < 4; c++) {
      let pivRow = c;
      let pivVal = Math.abs(A[4 * c + c]);
      for (let r = c + 1; r < 4; r++) {
        const v = Math.abs(A[4 * r + c]);
        if (v > pivVal) { pivVal = v; pivRow = r; }
      }
      if (pivVal < 1e-18 || !Number.isFinite(pivVal)) return false;

      if (pivRow !== c) {
        for (let k = 0; k < 4; k++) {
          const tmp = A[4 * c + k]; A[4 * c + k] = A[4 * pivRow + k]; A[4 * pivRow + k] = tmp;
          const tmp2 = out[4 * c + k]; out[4 * c + k] = out[4 * pivRow + k]; out[4 * pivRow + k] = tmp2;
        }
      }

      const piv = A[4 * c + c];
      const invP = 1 / piv;
      for (let k = 0; k < 4; k++) {
        A[4 * c + k] *= invP;
        out[4 * c + k] *= invP;
      }

      for (let r = 0; r < 4; r++) {
        if (r === c) continue;
        const f = A[4 * r + c];
        if (f === 0) continue;
        for (let k = 0; k < 4; k++) {
          A[4 * r + k] -= f * A[4 * c + k];
          out[4 * r + k] -= f * out[4 * c + k];
        }
      }
    }
    return true;
  }

  // ---------- Dynamics ----------
  function accelFromState(x, aOut) {
    const X = x[0], Y = x[2];
    const r2 = X * X + Y * Y;
    const r = Math.sqrt(Math.max(EPS, r2));
    const r3 = r2 * r;
    const k = -MU / Math.max(EPS, r3);
    aOut[0] = k * X;
    aOut[1] = k * Y;
    return aOut;
  }

  function dyn(x, uEci, aDist, out) {
    const a = dyn._a;
    accelFromState(x, a);
    const ax = a[0] + uEci[0] + aDist[0];
    const ay = a[1] + uEci[1] + aDist[1];
    out[0] = x[1];
    out[1] = ax;
    out[2] = x[3];
    out[3] = ay;
    return out;
  }
  dyn._a = new Float64Array(2);

  function rk4Step(x, uEci, aDist, dt, out) {
    const k1 = rk4Step._k1, k2 = rk4Step._k2, k3 = rk4Step._k3, k4 = rk4Step._k4;
    const xt = rk4Step._xt;

    dyn(x, uEci, aDist, k1);
    for (let i = 0; i < 4; i++) xt[i] = x[i] + 0.5 * dt * k1[i];

    dyn(xt, uEci, aDist, k2);
    for (let i = 0; i < 4; i++) xt[i] = x[i] + 0.5 * dt * k2[i];

    dyn(xt, uEci, aDist, k3);
    for (let i = 0; i < 4; i++) xt[i] = x[i] + dt * k3[i];

    dyn(xt, uEci, aDist, k4);

    for (let i = 0; i < 4; i++) {
      out[i] = x[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    return out;
  }
  rk4Step._k1 = new Float64Array(4);
  rk4Step._k2 = new Float64Array(4);
  rk4Step._k3 = new Float64Array(4);
  rk4Step._k4 = new Float64Array(4);
  rk4Step._xt = new Float64Array(4);

  function A_jacobian(x, outA) {
    const x1 = x[0], x3 = x[2];
    const r2 = x1 * x1 + x3 * x3;
    const r = Math.sqrt(Math.max(EPS, r2));
    const r5 = Math.pow(r, 5);
    const mu = MU;

    const a11 = mu * (2 * x1 * x1 - x3 * x3) / Math.max(EPS, r5);
    const a13 = 3 * mu * x1 * x3 / Math.max(EPS, r5);
    const a31 = a13;
    const a33 = mu * (2 * x3 * x3 - x1 * x1) / Math.max(EPS, r5);

    outA[0] = 0;   outA[1] = 1;   outA[2] = 0;   outA[3] = 0;
    outA[4] = a11; outA[5] = 0;   outA[6] = a13; outA[7] = 0;
    outA[8] = 0;   outA[9] = 0;   outA[10] = 0;  outA[11] = 1;
    outA[12] = a31;outA[13] = 0;  outA[14] = a33;outA[15] = 0;
    return outA;
  }

  function stationState(i, t, outPosVel) {
    const theta0 = (i) * Math.PI / 6;
    const ang = OMEGA_E * t + theta0;
    const c = Math.cos(ang), s = Math.sin(ang);
    const x = R_E * c;
    const y = R_E * s;
    const xd = -OMEGA_E * R_E * s;
    const yd = OMEGA_E * R_E * c;

    outPosVel[0] = x;
    outPosVel[1] = xd;
    outPosVel[2] = y;
    outPosVel[3] = yd;
    return outPosVel;
  }

  function visibleStations(xTrue, t, outIds) {
    const X = xTrue[0], Y = xTrue[2];
    let m = 0;
    const pv = visibleStations._pv;

    for (let i = 0; i < 12; i++) {
      stationState(i, t, pv);
      const sx = pv[0], sy = pv[2];
      const dx = X - sx;
      const dy = Y - sy;
      const dot = dx * sx + dy * sy;
      if (dot > 0) outIds[m++] = i;
    }
    return m;
  }
  visibleStations._pv = new Float64Array(4);

  function H_jacobian(x, t, visibleIds, m, yOut, HOut) {
    const x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    const pv = H_jacobian._pv;

    for (let k = 0; k < m; k++) {
      const i = visibleIds[k];
      stationState(i, t, pv);

      const Xi = pv[0], Xdi = pv[1], Yi = pv[2], Ydi = pv[3];
      const dX = x1 - Xi;
      const dXd = x2 - Xdi;
      const dY = x3 - Yi;
      const dYd = x4 - Ydi;

      const rho = Math.sqrt(Math.max(EPS, dX * dX + dY * dY));
      const a = dX * dXd + dY * dYd;
      const rhod = a / rho;
      const phi = Math.atan2(x3 - Yi, x1 - Xi);

      const idx = 3 * k;
      yOut[idx + 0] = rho;
      yOut[idx + 1] = rhod;
      yOut[idx + 2] = phi;

      const rho3 = rho * rho * rho;
      const rho2 = rho * rho;

      // rho
      HOut[(idx + 0) * 4 + 0] = dX / rho;
      HOut[(idx + 0) * 4 + 1] = 0;
      HOut[(idx + 0) * 4 + 2] = dY / rho;
      HOut[(idx + 0) * 4 + 3] = 0;

      // rhod
      HOut[(idx + 1) * 4 + 0] = (dXd / rho) - (a * dX / Math.max(EPS, rho3));
      HOut[(idx + 1) * 4 + 1] = dX / rho;
      HOut[(idx + 1) * 4 + 2] = (dYd / rho) - (a * dY / Math.max(EPS, rho3));
      HOut[(idx + 1) * 4 + 3] = dY / rho;

      // phi
      HOut[(idx + 2) * 4 + 0] = -dY / Math.max(EPS, rho2);
      HOut[(idx + 2) * 4 + 1] = 0;
      HOut[(idx + 2) * 4 + 2] = dX / Math.max(EPS, rho2);
      HOut[(idx + 2) * 4 + 3] = 0;
    }
  }
  H_jacobian._pv = new Float64Array(4);

  // ---------- Controller (LQR only) ----------
  const K_LQR_DEFAULT = new Float64Array([
    1.13202156e-4, 1.45525025e-2, -9.85215978e-4, 1.44003569e+1,
    1.07542370e-5, 2.15638768e-3, 1.00028071e-2, 1.92677533e+1
  ]);

  function eciToPolarDeviations(x, t, ref) {
    const X = x[0], Xd = x[1], Y = x[2], Yd = x[3];
    const r2 = X * X + Y * Y;
    const r = Math.sqrt(Math.max(EPS, r2));
    const theta = Math.atan2(Y, X);
    const rdot = (X * Xd + Y * Yd) / Math.max(EPS, r);
    const thetadot = (X * Yd - Y * Xd) / Math.max(EPS, r2);

    const thetaRef = ref.theta0 + ref.n0 * t;

    const dr = r - ref.r0;
    const drdot = rdot;
    const dtheta = wrapPi(theta - thetaRef);
    const dthetadot = thetadot - ref.n0;

    return { theta, dr, drdot, dtheta, dthetadot };
  }

  function radialTangentialToEci(theta, ur, ut, out) {
    const c = Math.cos(theta), s = Math.sin(theta);
    const erx = c, ery = s;
    const etx = -s, ety = c;
    out[0] = ur * erx + ut * etx;
    out[1] = ur * ery + ut * ety;
    return out;
  }

  function controlLawLqr(xCtrl, outUrUt) {
    const K = K_LQR_DEFAULT;
    const ur = -(K[0] * xCtrl[0] + K[1] * xCtrl[1] + K[2] * xCtrl[2] + K[3] * xCtrl[3]);
    const ut = -(K[4] * xCtrl[0] + K[5] * xCtrl[1] + K[6] * xCtrl[2] + K[7] * xCtrl[3]);
    outUrUt[0] = ur;
    outUrUt[1] = ut;
    return outUrUt;
  }

  function saturateU(ur, ut, umax, out) {
    const mag = Math.hypot(ur, ut);
    if (!Number.isFinite(mag) || mag <= umax) {
      out[0] = ur;
      out[1] = ut;
      out[2] = mag;
      out[3] = 0;
      return out;
    }
    const s = umax / Math.max(EPS, mag);
    out[0] = ur * s;
    out[1] = ut * s;
    out[2] = umax;
    out[3] = 1;
    return out;
  }

  // ---------- EKF ----------
  function ekfPredict(xhat, P, uEci, Q, dt) {
    const aDist = ekfPredict._zero2;
    const xnew = ekfPredict._xnew;

    rk4Step(xhat, uEci, aDist, dt, xnew);
    for (let i = 0; i < 4; i++) xhat[i] = xnew[i];

    const A = ekfPredict._A;
    A_jacobian(xhat, A);

    const F = ekfPredict._F;
    mat4Identity(F);
    for (let i = 0; i < 16; i++) F[i] += dt * A[i];

    const FP = ekfPredict._FP;
    mat4Mul(FP, F, P);

    const FPFt = ekfPredict._FPFt;
    mat4MulT(FPFt, FP, F);

    const dt2 = dt * dt;
    const q11 = Q[0], q12 = Q[1], q21 = Q[2], q22 = Q[3];
    FPFt[5] += dt2 * q11;
    FPFt[7] += dt2 * q12;
    FPFt[13] += dt2 * q21;
    FPFt[15] += dt2 * q22;

    mat4Copy(P, FPFt);
    mat4Symmetrize(P);
    mat4AddDiag(P, 1e-14);
  }
  ekfPredict._A = new Float64Array(16);
  ekfPredict._F = new Float64Array(16);
  ekfPredict._FP = new Float64Array(16);
  ekfPredict._FPFt = new Float64Array(16);
  ekfPredict._zero2 = new Float64Array([0, 0]);
  ekfPredict._xnew = new Float64Array(4);

  function ekfUpdate(xhat, P, yMeas, t, visIds, m, Rper, outDiag) {
    const yhat = ekfUpdate._yhat;
    const H = ekfUpdate._H;
    H_jacobian(xhat, t, visIds, m, yhat, H);

    const n = 3 * m;
    if (n === 0) {
      outDiag.innovNorm = 0;
      outDiag.nis = NaN;
      outDiag.sSingular = false;
      return;
    }

    const v = ekfUpdate._v;
    let innovNorm2 = 0;
    for (let i = 0; i < n; i++) {
      let di = yMeas[i] - yhat[i];
      if ((i % 3) === 2) di = wrapPi(di);
      v[i] = di;
      innovNorm2 += di * di;
    }
    outDiag.innovNorm = Math.sqrt(Math.max(0, innovNorm2));

    const R = ekfUpdate._R;
    for (let i = 0; i < n * n; i++) R[i] = 0;

    const r00 = Rper[0], r11 = Rper[4], r22 = Rper[8];
    for (let k = 0; k < m; k++) {
      const base = 3 * k;
      R[(base + 0) * n + (base + 0)] = r00;
      R[(base + 1) * n + (base + 1)] = r11;
      R[(base + 2) * n + (base + 2)] = r22;
    }

    const HP = ekfUpdate._HP;
    for (let i = 0; i < n; i++) {
      const h0 = H[i * 4 + 0], h1 = H[i * 4 + 1], h2 = H[i * 4 + 2], h3 = H[i * 4 + 3];
      HP[i * 4 + 0] = h0 * P[0] + h1 * P[4] + h2 * P[8] + h3 * P[12];
      HP[i * 4 + 1] = h0 * P[1] + h1 * P[5] + h2 * P[9] + h3 * P[13];
      HP[i * 4 + 2] = h0 * P[2] + h1 * P[6] + h2 * P[10] + h3 * P[14];
      HP[i * 4 + 3] = h0 * P[3] + h1 * P[7] + h2 * P[11] + h3 * P[15];
    }

    const S = ekfUpdate._S;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const s = HP[i * 4 + 0] * H[j * 4 + 0] +
                  HP[i * 4 + 1] * H[j * 4 + 1] +
                  HP[i * 4 + 2] * H[j * 4 + 2] +
                  HP[i * 4 + 3] * H[j * 4 + 3];
        S[i * n + j] = s + R[i * n + j];
      }
    }

    const Swork = ekfUpdate._Swork;
    for (let i = 0; i < n * n; i++) Swork[i] = S[i];

    let ok = cholDecompInPlace(Swork, n);
    let jitter = 1e-12;
    let tries = 0;

    while (!ok && tries < 6) {
      for (let d = 0; d < n; d++) Swork[d * n + d] = S[d * n + d] + jitter;
      for (let i = 0; i < n * n; i++) {
        if ((i % (n + 1)) !== 0) Swork[i] = S[i];
      }
      ok = cholDecompInPlace(Swork, n);
      jitter *= 10;
      tries++;
    }

    outDiag.sSingular = !ok;
    if (!ok) {
      outDiag.nis = NaN;
      return;
    }

    const tmp = ekfUpdate._tmpn;
    for (let i = 0; i < n; i++) tmp[i] = v[i];
    cholSolve(Swork, n, tmp, tmp);

    let nis = 0;
    for (let i = 0; i < n; i++) nis += v[i] * tmp[i];
    outDiag.nis = nis;

    // W = S^-1 * (H P)  (n x 4)
    const W = ekfUpdate._W;
    for (let col = 0; col < 4; col++) {
      for (let i = 0; i < n; i++) tmp[i] = HP[i * 4 + col];
      cholSolve(Swork, n, tmp, tmp);
      for (let i = 0; i < n; i++) W[i * 4 + col] = tmp[i];
    }

    // K = P H^T S^-1 = (W^T)  (4 x n)
    const K = ekfUpdate._K;
    for (let i = 0; i < n; i++) {
      K[0 * n + i] = W[i * 4 + 0];
      K[1 * n + i] = W[i * 4 + 1];
      K[2 * n + i] = W[i * 4 + 2];
      K[3 * n + i] = W[i * 4 + 3];
    }

    const dx = ekfUpdate._dx4;
    dx[0] = 0; dx[1] = 0; dx[2] = 0; dx[3] = 0;
    for (let i = 0; i < n; i++) {
      const vi = v[i];
      dx[0] += K[0 * n + i] * vi;
      dx[1] += K[1 * n + i] * vi;
      dx[2] += K[2 * n + i] * vi;
      dx[3] += K[3 * n + i] * vi;
    }
    for (let i = 0; i < 4; i++) xhat[i] += dx[i];

    // P <- (I - K H) P   (simple form)
    const KH = ekfUpdate._KH;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += K[r * n + i] * H[i * 4 + c];
        KH[r * 4 + c] = s;
      }
    }

    const IminusKH = ekfUpdate._ImKH;
    mat4Identity(IminusKH);
    for (let i = 0; i < 16; i++) IminusKH[i] -= KH[i];

    const newP = ekfUpdate._newP;
    mat4Mul(newP, IminusKH, P);
    mat4Copy(P, newP);
    mat4Symmetrize(P);
    mat4AddDiag(P, 1e-14);
  }
  ekfUpdate._yhat = new Float64Array(36);
  ekfUpdate._H = new Float64Array(36 * 4);
  ekfUpdate._v = new Float64Array(36);
  ekfUpdate._R = new Float64Array(36 * 36);
  ekfUpdate._HP = new Float64Array(36 * 4);
  ekfUpdate._S = new Float64Array(36 * 36);
  ekfUpdate._Swork = new Float64Array(36 * 36);
  ekfUpdate._W = new Float64Array(36 * 4);
  ekfUpdate._K = new Float64Array(4 * 36);
  ekfUpdate._tmpn = new Float64Array(36);
  ekfUpdate._dx4 = new Float64Array(4);
  ekfUpdate._KH = new Float64Array(16);
  ekfUpdate._ImKH = new Float64Array(16);
  ekfUpdate._newP = new Float64Array(16);

  // ---------- Ring buffer ----------
  class RingBuffer {
    constructor(cap, dim) {
      this.cap = cap;
      this.dim = dim;
      this.data = new Float64Array(cap * dim);
      this.t = new Float64Array(cap);
      this.idx = 0;
      this.len = 0;
    }
    push(time, vec) {
      const i = this.idx;
      this.t[i] = time;
      const off = i * this.dim;
      for (let d = 0; d < this.dim; d++) this.data[off + d] = vec[d];
      this.idx = (i + 1) % this.cap;
      this.len = Math.min(this.cap, this.len + 1);
    }
    forEachChronological(cb) {
      const n = this.len;
      const start = (this.idx - n + this.cap) % this.cap;
      for (let k = 0; k < n; k++) {
        const i = (start + k) % this.cap;
        cb(this.t[i], this.data, i * this.dim, k, n);
      }
    }
    lastTime() {
      if (this.len === 0) return 0;
      return this.t[(this.idx - 1 + this.cap) % this.cap];
    }
  }

  // ---------- Plot Canvas ----------
  function niceStep(span, targetTicks) {
    const s = Math.max(EPS, span);
    const raw = s / Math.max(2, targetTicks);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step = 1;
    if (norm < 1.5) step = 1;
    else if (norm < 3.5) step = 2;
    else if (norm < 7.5) step = 5;
    else step = 10;
    return step * mag;
  }

  function makeTicks(min, max, target) {
    const span = max - min;
    const step = niceStep(span, target);
    const t0 = Math.ceil(min / step) * step;
    const t1 = Math.floor(max / step) * step;
    const out = [];
    for (let v = t0; v <= t1 + 0.5 * step; v += step) {
      out.push(v);
      if (out.length > 200) break;
    }
    return { step, ticks: out };
  }

  class PlotCanvas {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d', { alpha: true });
      this.opts = opts || {};
      this.dpr = 1;
      this.w = 0;
      this.h = 0;

      this.xPan = 0;
      this.xZoom = 1;
      this.yPan = 0;
      this.yZoom = 1;

      this.isDragging = false;
      this.dragStart = { x: 0, y: 0, xPan: 0, yPan: 0 };
      this.hover = { x: 0, y: 0, on: false };

      this._bind();
    }

    _bind() {
      const c = this.canvas;

      c.addEventListener('mousemove', (e) => {
        const r = c.getBoundingClientRect();
        this.hover.x = (e.clientX - r.left) * (window.devicePixelRatio || 1);
        this.hover.y = (e.clientY - r.top) * (window.devicePixelRatio || 1);
        this.hover.on = true;
      });

      c.addEventListener('mouseleave', () => { this.hover.on = false; });

      c.addEventListener('mousedown', (e) => {
        if (!app.state.running) {
          this.isDragging = true;
          this.dragStart.x = e.clientX;
          this.dragStart.y = e.clientY;
          this.dragStart.xPan = this.xPan;
          this.dragStart.yPan = this.yPan;
        }
      });

      window.addEventListener('mouseup', () => { this.isDragging = false; });

      window.addEventListener('mousemove', (e) => {
        if (this.isDragging && !app.state.running) {
          const dx = e.clientX - this.dragStart.x;
          const dy = e.clientY - this.dragStart.y;

          const span = this._xSpan();
          const xs = (span / Math.max(1, this.w)) * (1 / Math.max(EPS, this.xZoom));
          this.xPan = this.dragStart.xPan - dx * xs;

          const ys = (this._ySpanPx / Math.max(1, this.h)) * (1 / Math.max(EPS, this.yZoom));
          this.yPan = this.dragStart.yPan + dy * ys;
        }
      });

      c.addEventListener('wheel', (e) => {
        if (!app.state.running) {
          e.preventDefault();
          const factor = (e.deltaY < 0) ? 1.12 : 0.89;
          this.xZoom = clamp(this.xZoom * factor, 0.2, 80);
          this.yZoom = clamp(this.yZoom * factor, 0.2, 80);
        }
      }, { passive: false });
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (w !== this.canvas.width || h !== this.canvas.height) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.dpr = dpr;
      this.w = w;
      this.h = h;
    }

    _xSpan() {
      if (typeof this.opts.xSpanFn === 'function') return Math.max(10, this.opts.xSpanFn());
      return this.opts.xSpan || 21600;
    }

    _nicePad(ymin, ymax) {
      if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) return [-1, 1];
      const span = Math.abs(ymax - ymin);
      if (span < 1e-12) {
        const d = Math.max(1, Math.abs(ymin) * 0.1);
        return [ymin - d, ymax + d];
      }
      const pad = span * 0.10;
      let lo = ymin - pad;
      let hi = ymax + pad;
      const minSpan = (Number.isFinite(this.opts.minYSpan) ? this.opts.minYSpan : null);
      if (minSpan && (hi - lo) < minSpan) {
        const c = 0.5 * (hi + lo);
        lo = c - 0.5 * minSpan;
        hi = c + 0.5 * minSpan;
      }
      return [lo, hi];
    }

    _legendItems(series) {
      if (Array.isArray(this.opts.legendItems)) return this.opts.legendItems;
      const items = [];
      const seen = new Set();
      for (const s of series) {
        if (!s || !s.name) continue;
        if (s.legend === false) continue;
        if (seen.has(s.name)) continue;
        seen.add(s.name);
        items.push({ label: s.name, color: s.color, type: s.type });
        if (items.length >= 6) break;
      }
      return items;
    }

    draw(series) {
      this.resize();
      const ctx = this.ctx;
      const W = this.w, H = this.h;
      const dpr = this.dpr;

      const padL = 54 * dpr, padR = 14 * dpr, padT = 12 * dpr, padB = 36 * dpr;
      const px0 = padL, py0 = padT;
      const pw = Math.max(1, W - padL - padR);
      const ph = Math.max(1, H - padT - padB);

      ctx.save();
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.fillRect(0, 0, W, H);

      const tNowRaw = app.hist.truth.lastTime();
      const tNow = Math.max(0, Number.isFinite(tNowRaw) ? tNowRaw : 0);
      const span = this._xSpan();

      let base0, base1;
      if (tNow < span) {
        base0 = 0;
        base1 = Math.max(1e-6, tNow);
        if (base1 < 1) base1 = 1;
      } else {
        base0 = tNow - span;
        base1 = tNow;
      }

      let x0 = base0 + this.xPan;
      let x1 = base1 + this.xPan;
      if (x1 <= x0 + 1e-6) x1 = x0 + 1;

      const xc = 0.5 * (x0 + x1);
      const half = 0.5 * (x1 - x0) / Math.max(EPS, this.xZoom);
      x0 = xc - half;
      x1 = xc + half;

      if (x0 < 0) {
        const w = (x1 - x0);
        x0 = 0;
        x1 = x0 + Math.max(1, w);
      }

      let ymin = Infinity, ymax = -Infinity;
      let fixedY = null;
      if (Array.isArray(this.opts.fixedY)) fixedY = this.opts.fixedY;
      if (typeof this.opts.fixedYFn === 'function') fixedY = this.opts.fixedYFn();

      if (fixedY) {
        ymin = fixedY[0];
        ymax = fixedY[1];
      } else {
        for (const s of series) {
          if (!s || s.type === 'hline' || s.type === 'band') continue;
          s.scanRange(x0, x1, (y) => {
            if (Number.isFinite(y)) {
              ymin = Math.min(ymin, y);
              ymax = Math.max(ymax, y);
            }
          });
        }
        if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) { ymin = -1; ymax = 1; }
        const padded = this._nicePad(ymin, ymax);
        ymin = padded[0];
        ymax = padded[1];

        const yc = 0.5 * (ymin + ymax) + this.yPan;
        const hy = 0.5 * (ymax - ymin) / Math.max(EPS, this.yZoom);
        ymin = yc - hy;
        ymax = yc + hy;
      }

      if (!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymax <= ymin + 1e-12) {
        ymin = -1; ymax = 1;
      }

      this._ySpanPx = (ymax - ymin);

      const xToPx = (t) => px0 + ((t - x0) / (x1 - x0)) * pw;
      const yToPx = (y) => py0 + (1 - (y - ymin) / (ymax - ymin)) * ph;

      ctx.save();
      ctx.beginPath();
      ctx.rect(px0, py0, pw, ph);
      ctx.clip();

      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.07)';
      const xt = makeTicks(x0, x1, 7);
      const yt = makeTicks(ymin, ymax, 5);

      for (const xv of xt.ticks) {
        const x = xToPx(xv);
        ctx.beginPath();
        ctx.moveTo(x, py0);
        ctx.lineTo(x, py0 + ph);
        ctx.stroke();
      }
      for (const yv of yt.ticks) {
        const y = yToPx(yv);
        ctx.beginPath();
        ctx.moveTo(px0, y);
        ctx.lineTo(px0 + pw, y);
        ctx.stroke();
      }

      for (const s of series) {
        if (!s) continue;
        s.draw(ctx, xToPx, yToPx, x0, x1, ymin, ymax, px0, py0, pw, ph, dpr);
      }

      if (!app.state.running && this.hover.on) {
        const mx = this.hover.x;
        const my = this.hover.y;
        if (mx >= px0 && mx <= px0 + pw && my >= py0 && my <= py0 + ph) {
          ctx.strokeStyle = 'rgba(255,255,255,0.14)';
          ctx.beginPath();
          ctx.moveTo(mx, py0);
          ctx.lineTo(mx, py0 + ph);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(px0, my);
          ctx.lineTo(px0 + pw, my);
          ctx.stroke();

          const t = x0 + ((mx - px0) / pw) * (x1 - x0);
          const y = ymax - ((my - py0) / ph) * (ymax - ymin);

          ctx.fillStyle = 'rgba(0,0,0,0.52)';
          ctx.fillRect(px0 + 6 * dpr, py0 + 6 * dpr, Math.min(pw - 12 * dpr, 186 * dpr), 40 * dpr);

          ctx.fillStyle = 'rgba(255,255,255,0.92)';
          ctx.font = `${12 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
          ctx.fillText(`t=${t.toFixed(2)}s`, px0 + 12 * dpr, py0 + 22 * dpr);
          ctx.fillText(
            `y=${Number.isFinite(y) ? y.toExponential(3) : '–'}`,
            px0 + 12 * dpr,
            py0 + 38 * dpr
          );
        }
      }

      ctx.restore();

      ctx.strokeStyle = 'rgba(255,255,255,0.20)';
      ctx.lineWidth = 1;
      ctx.strokeRect(px0, py0, pw, ph);

      // axes labels
      ctx.fillStyle = 'rgba(255,255,255,0.82)';
      ctx.font = `${11 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textBaseline = 'middle';

      for (const xv of xt.ticks) {
        const x = xToPx(xv);
        ctx.strokeStyle = 'rgba(255,255,255,0.22)';
        ctx.beginPath();
        ctx.moveTo(x, py0 + ph);
        ctx.lineTo(x, py0 + ph + 4 * dpr);
        ctx.stroke();
        const txt = (Math.abs(xv) >= 10000) ? `${Math.round(xv)}` : `${xv.toFixed(0)}`;
        ctx.fillText(txt, x - 10 * dpr, py0 + ph + 14 * dpr);
      }

      ctx.textAlign = 'right';
      for (const yv of yt.ticks) {
        const y = yToPx(yv);
        ctx.strokeStyle = 'rgba(255,255,255,0.22)';
        ctx.beginPath();
        ctx.moveTo(px0 - 4 * dpr, y);
        ctx.lineTo(px0, y);
        ctx.stroke();

        let txt = `${yv.toFixed(2)}`;
        const ay = Math.abs(yv);
        if (ay >= 1000) txt = `${Math.round(yv)}`;
        else if (ay >= 10) txt = `${yv.toFixed(1)}`;
        ctx.fillText(txt, px0 - 8 * dpr, y);
      }

      ctx.textAlign = 'left';
      const xLabel = this.opts.xLabel || 't (s)';
      const yLabel = this.opts.yLabel || '';

      ctx.fillStyle = 'rgba(255,255,255,0.78)';
      ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(xLabel, px0 + pw * 0.5, H - 8 * dpr);

      if (yLabel) {
        ctx.save();
        ctx.translate(14 * dpr, py0 + ph * 0.5);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
      }

      // legend
      const items = this._legendItems(series);
      if (items && items.length) {
        const lx = px0 + 8 * dpr;
        const ly = py0 + 8 * dpr;
        const lh = 16 * dpr;
        const pad = 8 * dpr;
        const maxText = items.reduce((m, it) => Math.max(m, (it.label || '').length), 0);
        const boxW = clamp((pad * 2 + (maxText * 7 * dpr) + 44 * dpr), 120 * dpr, 240 * dpr);
        const boxH = pad * 2 + items.length * lh;

        ctx.fillStyle = 'rgba(0,0,0,0.52)';
        ctx.fillRect(lx, ly, boxW, boxH);
        ctx.strokeStyle = 'rgba(255,255,255,0.16)';
        ctx.strokeRect(lx, ly, boxW, boxH);

        ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';

        for (let i = 0; i < items.length; i++) {
          const it = items[i];
          const y = ly + pad + (i + 0.5) * lh;
          const swx = lx + pad;

          ctx.strokeStyle = it.color || 'rgba(255,255,255,0.8)';
          ctx.fillStyle = it.color || 'rgba(255,255,255,0.8)';

          if (it.type === 'dots') {
            ctx.beginPath();
            ctx.arc(swx + 7 * dpr, y, 3 * dpr, 0, TWO_PI);
            ctx.fill();
          } else {
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(swx, y);
            ctx.lineTo(swx + 16 * dpr, y);
            ctx.stroke();
          }

          ctx.fillStyle = 'rgba(255,255,255,0.88)';
          ctx.fillText(it.label, swx + 24 * dpr, y);
        }
      }

      ctx.restore();
    }
  }

  // Series factories (no decimation)
  function makeLineSeries(buf, dimIndex, color, label) {
    return {
      name: label,
      type: 'line',
      color,
      scanRange: (t0, t1, acc) => {
        buf.forEachChronological((t, data, off) => {
          if (t >= t0 && t <= t1) acc(data[off + dimIndex]);
        });
      },
      draw: (ctx, xToPx, yToPx, t0, t1) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        buf.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const y = data[off + dimIndex];
          if (!Number.isFinite(y)) return;
          const x = xToPx(t);
          const py = yToPx(y);
          if (!started) { ctx.moveTo(x, py); started = true; }
          else ctx.lineTo(x, py);
        });
        ctx.stroke();
      }
    };
  }

  function makeGappedLineSeries(buf, dimIndex, color, label, gapMultiplier = 1.75) {
    return {
      name: label,
      type: 'line',
      color,
      scanRange: (t0, t1, acc) => {
        buf.forEachChronological((t, data, off) => {
          if (t >= t0 && t <= t1) acc(data[off + dimIndex]);
        });
      },
      draw: (ctx, xToPx, yToPx, t0, t1) => {
        const gapThresh = Math.max(1, app.state.dt) * gapMultiplier;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        let started = false;
        let lastT = NaN;

        buf.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const y = data[off + dimIndex];
          if (!Number.isFinite(y)) return;

          const x = xToPx(t);
          const py = yToPx(y);

          if (!started) {
            ctx.moveTo(x, py);
            started = true;
          } else {
            if (Number.isFinite(lastT) && (t - lastT) > gapThresh) {
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(x, py);
            } else {
              ctx.lineTo(x, py);
            }
          }
          lastT = t;
        });

        ctx.stroke();
      }
    };
  }

  function makeScaledLineSeries(buf, dimIndex, scale, color, label) {
    return {
      name: label,
      type: 'line',
      color,
      scanRange: (t0, t1, acc) => {
        buf.forEachChronological((t, data, off) => {
          if (t >= t0 && t <= t1) acc(scale * data[off + dimIndex]);
        });
      },
      draw: (ctx, xToPx, yToPx, t0, t1, _ymin, _ymax, _px0, _py0, _pw, _ph, dpr) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.7;
        ctx.setLineDash([6 * dpr, 6 * dpr]);
        ctx.beginPath();

        let started = false;
        buf.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const y = scale * data[off + dimIndex];
          if (!Number.isFinite(y)) return;
          const x = xToPx(t);
          const py = yToPx(y);
          if (!started) { ctx.moveTo(x, py); started = true; }
          else ctx.lineTo(x, py);
        });

        ctx.stroke();
        ctx.setLineDash([]);
      }
    };
  }

  function makeDotsSeries(buf, dimIndex, color, label) {
    return {
      name: label,
      type: 'dots',
      color,
      scanRange: (t0, t1, acc) => {
        buf.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const y = data[off + dimIndex];
          if (Number.isFinite(y)) acc(y);
        });
      },
      draw: (ctx, xToPx, yToPx, t0, t1, _ymin, _ymax, _px0, _py0, _pw, _ph, dpr) => {
        ctx.fillStyle = color;
        const r = Math.max(1.5, 2.2 * dpr);
        buf.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const y = data[off + dimIndex];
          if (!Number.isFinite(y)) return;
          const x = xToPx(t);
          const py = yToPx(y);
          ctx.beginPath();
          ctx.arc(x, py, Math.max(0.8, r), 0, TWO_PI);
          ctx.fill();
        });
      }
    };
  }

  function makeHLine(yValFn, color, label) {
    return {
      name: label,
      type: 'hline',
      color,
      scanRange: () => { },
      draw: (ctx, _xToPx, yToPx, _t0, _t1, _ymin, _ymax, px0, py0, pw, ph, dpr) => {
        const yVal = (typeof yValFn === 'function') ? yValFn() : yValFn;
        if (!Number.isFinite(yVal)) return;
        const py = yToPx(yVal);
        if (py < py0 || py > py0 + ph) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([7 * dpr, 7 * dpr]);
        ctx.beginPath();
        ctx.moveTo(px0, py);
        ctx.lineTo(px0 + pw, py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };
  }

  // ---------- Orbit view ----------

  // ---------- Relative Position + Covariance Ellipse (error-space) ----------
  // Plots (x_true - x̂, y_true - ŷ) and overlays 1σ/2σ/3σ ellipses derived from the EKF
  // position covariance submatrix.
  //
  // For a 2D zero-mean Gaussian, the probability mass inside an ellipse at k-σ is:
  //   P = 1 - exp(-k^2 / 2)
  // so 1σ ≈ 39.3%, 2σ ≈ 86.5%, 3σ ≈ 98.9%.
  class RelPosCovPlot {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d', { alpha: true });
      this.dpr = 1;
      this.w = 0;
      this.h = 0;

      // Pan in km (error-space) and zoom factor
      this.panX = 0;
      this.panY = 0;
      this.zoom = 1;

      this.isDragging = false;
      this.dragStart = { x: 0, y: 0, panX: 0, panY: 0 };

      this._halfRange = 1; // km, smoothed view half-span

      // Short fading trail settings (kept internal; no UI changes)
      this._trailMaxSec = 900;   // show last N seconds (keeps it readable)
      this._trailMaxPts = 420;   // cap segments for performance

      this._bind();
    }

    _bind() {
      const c = this.canvas;

      c.addEventListener('mousedown', (e) => {
        if (app.state.running) return;
        this.isDragging = true;
        this.dragStart.x = e.clientX;
        this.dragStart.y = e.clientY;
        this.dragStart.panX = this.panX;
        this.dragStart.panY = this.panY;
      });

      window.addEventListener('mouseup', () => { this.isDragging = false; });

      window.addEventListener('mousemove', (e) => {
        if (!this.isDragging || app.state.running) return;
        const rect = c.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const W = Math.max(1, rect.width * dpr);
        const H = Math.max(1, rect.height * dpr);

        const pad = 14 * dpr;
        const spanPx = Math.max(1, Math.min(W, H) - 2 * pad);
        const scale = (spanPx / (2 * Math.max(1e-9, this._halfRange))) * this.zoom; // px per km

        const dxPx = (e.clientX - this.dragStart.x) * dpr;
        const dyPx = (e.clientY - this.dragStart.y) * dpr;

        this.panX = this.dragStart.panX - dxPx / Math.max(1e-9, scale);
        this.panY = this.dragStart.panY + dyPx / Math.max(1e-9, scale);
      });

      c.addEventListener('wheel', (e) => {
        if (app.state.running) return;
        e.preventDefault();
        const factor = (e.deltaY < 0) ? 1.12 : 0.89;
        this.zoom = clamp(this.zoom * factor, 0.25, 50);
      }, { passive: false });
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (w !== this.canvas.width || h !== this.canvas.height) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.dpr = dpr;
      this.w = w;
      this.h = h;
    }

    _posCovEigen(P) {
      // Position submatrix (X, Y) in row-major 4x4 is [0,2; 8,10]
      const px = P[0], py = P[10], pxy = P[2];
      const tr = px + py;
      const det = px * py - pxy * pxy;
      const disc = Math.max(0, tr * tr * 0.25 - det);
      const s = Math.sqrt(disc);

      let l1 = tr * 0.5 + s;
      let l2 = tr * 0.5 - s;
      l1 = Math.max(1e-18, l1);
      l2 = Math.max(1e-18, l2);

      const ang = 0.5 * Math.atan2(2 * pxy, px - py);
      return { l1, l2, ang };
    }

    _drawLegend(ctx, pad, dpr, items) {
      if (!items || !items.length) return;

      const lh = 16 * dpr;
      const boxPad = 8 * dpr;

      const maxText = items.reduce((m, it) => Math.max(m, (it.label || '').length), 0);
      const boxW = clamp((boxPad * 2 + (maxText * 7.2 * dpr) + 44 * dpr), 160 * dpr, 280 * dpr);
      const boxH = boxPad * 2 + items.length * lh;

      // place below the Δx label so we don't overlap it
      const lx = pad;
      const ly = pad + 18 * dpr;

      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(lx, ly, boxW, boxH);
      ctx.strokeStyle = 'rgba(255,255,255,0.16)';
      ctx.strokeRect(lx, ly, boxW, boxH);

      ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';

      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        const y = ly + boxPad + (i + 0.5) * lh;
        const swx = lx + boxPad;

        ctx.strokeStyle = it.color || 'rgba(255,255,255,0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(swx, y);
        ctx.lineTo(swx + 16 * dpr, y);
        ctx.stroke();

        ctx.fillStyle = 'rgba(255,255,255,0.88)';
        ctx.fillText(it.label, swx + 24 * dpr, y);
      }
    }

    draw() {
      if (!this.canvas) return;

      this.resize();
      const ctx = this.ctx;
      const W = this.w, H = this.h;
      const dpr = this.dpr;

      const pad = 14 * dpr;
      const cx = W * 0.5;
      const cy = H * 0.5;

      ctx.save();
      ctx.clearRect(0, 0, W, H);

      // background
      ctx.fillStyle = 'rgba(0,0,0,0.20)';
      ctx.fillRect(0, 0, W, H);

      const s = app.state;
      const liveT = getLiveSimTime();
      const span = Math.max(60, s.plotSpanSec || 10800);
      const t0 = Math.max(0, liveT - span);
      const t1 = liveT;

      // compute view extents from recent error trace + 3σ ellipse
      let maxAbs = 0;
      const err = app.hist.errComp;
      if (err && err.len > 0) {
        err.forEachChronological((t, data, off) => {
          if (t < t0 || t > t1) return;
          const ex = data[off + 0];
          const ey = data[off + 1];
          if (Number.isFinite(ex)) maxAbs = Math.max(maxAbs, Math.abs(ex));
          if (Number.isFinite(ey)) maxAbs = Math.max(maxAbs, Math.abs(ey));
        });
      }

      const eig = this._posCovEigen(s.P);
      const sigMax = Math.sqrt(Math.max(eig.l1, eig.l2));
      const wantHalf = Math.max(1e-3, maxAbs, 3.0 * sigMax) * 1.25;

      // smooth to prevent flicker (fast expand, slow contract)
      if (!Number.isFinite(this._halfRange) || this._halfRange <= 0) this._halfRange = wantHalf;
      if (wantHalf > this._halfRange) this._halfRange = wantHalf;
      else this._halfRange = 0.96 * this._halfRange + 0.04 * wantHalf;

      const spanPx = Math.max(1, Math.min(W, H) - 2 * pad);
      const pxPerKm = (spanPx / (2 * Math.max(1e-9, this._halfRange))) * this.zoom;

      const xToPx = (x) => cx + (x - this.panX) * pxPerKm;
      const yToPx = (y) => cy - (y - this.panY) * pxPerKm;

      // grid
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      const gridCount = 4;
      for (let i = -gridCount; i <= gridCount; i++) {
        const gx = cx + i * (spanPx / (2 * gridCount));
        const gy = cy + i * (spanPx / (2 * gridCount));
        ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, H - pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
      }

      // axes
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad, yToPx(0)); ctx.lineTo(W - pad, yToPx(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(xToPx(0), pad); ctx.lineTo(xToPx(0), H - pad); ctx.stroke();

      // ----- error trail (short + fading) -----
      const trailSec = Math.min(span, this._trailMaxSec);
      const tTrail0 = Math.max(t0, t1 - trailSec);

      if (err && err.len > 1) {
        const expected = Math.max(1, Math.ceil(trailSec / Math.max(1e-6, s.dt)));
        const stride = Math.max(1, Math.ceil(expected / Math.max(40, this._trailMaxPts)));

        let n = 0;
        let kIn = 0;

        const pts = RelPosCovPlot._trailPts;
        const cap = RelPosCovPlot._trailPtsCap;

        err.forEachChronological((t, data, off) => {
          if (t < tTrail0 || t > t1) return;
          if ((kIn++ % stride) !== 0) return;

          const ex = data[off + 0];
          const ey = data[off + 1];
          if (!Number.isFinite(ex) || !Number.isFinite(ey)) return;

          if (n < cap) {
            pts[2 * n + 0] = xToPx(ex);
            pts[2 * n + 1] = yToPx(ey);
            n++;
          }
        });

        if (n > 1) {
          ctx.save();
          ctx.strokeStyle = 'rgba(255,122,24,1)';
          ctx.lineWidth = 2;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          const gamma = 1.8;     // how quickly it fades
          const aMax = 0.85;     // newest segment alpha
          const aMin = 0.06;     // oldest segment alpha (still barely visible)

          for (let i = 1; i < n; i++) {
            const f = i / (n - 1);
            const a = aMin + (aMax - aMin) * Math.pow(f, gamma);

            ctx.globalAlpha = a;
            ctx.beginPath();
            ctx.moveTo(pts[2 * (i - 1) + 0], pts[2 * (i - 1) + 1]);
            ctx.lineTo(pts[2 * i + 0], pts[2 * i + 1]);
            ctx.stroke();
          }

          ctx.restore();
        }
      }

      // current error point (last sample)
      let exNow = NaN, eyNow = NaN;
      if (err && err.len > 0) {
        const li = (err.idx - 1 + err.cap) % err.cap;
        const off = li * err.dim;
        exNow = err.data[off + 0];
        eyNow = err.data[off + 1];
      }
      if (Number.isFinite(exNow) && Number.isFinite(eyNow)) {
        ctx.fillStyle = 'rgba(255,122,24,0.95)';
        ctx.beginPath();
        ctx.arc(xToPx(exNow), yToPx(eyNow), Math.max(2.5, 3.0 * dpr), 0, TWO_PI);
        ctx.fill();
      }

      // 1σ/2σ/3σ ellipses centered at origin (distinct colors)
      const rings = [
        { k: 1, color: 'rgba(34,197,94,0.62)' },   // green
        { k: 2, color: 'rgba(59,130,246,0.62)' },  // blue
        { k: 3, color: 'rgba(168,85,247,0.62)' }   // violet
      ];

      for (let i = 0; i < rings.length; i++) {
        const k = rings[i].k;
        const a = k * Math.sqrt(eig.l1);
        const b = k * Math.sqrt(eig.l2);
        if (!Number.isFinite(a) || !Number.isFinite(b)) continue;

        ctx.save();
        // draw in "world" coords with y-up by using a flipped scale
        ctx.translate(cx - this.panX * pxPerKm, cy + this.panY * pxPerKm);
        ctx.scale(pxPerKm, -pxPerKm);
        ctx.rotate(eig.ang);

        ctx.strokeStyle = rings[i].color;
        ctx.lineWidth = (1.6 * dpr) / Math.max(1e-9, pxPerKm);
        ctx.beginPath();
        ctx.ellipse(0, 0, a, b, 0, 0, TWO_PI);
        ctx.stroke();

        ctx.restore();
      }

      // labels
      ctx.fillStyle = 'rgba(255,255,255,0.78)';
      ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      ctx.fillText('Δx (km)', pad, pad - 1 * dpr);

      ctx.save();
      ctx.translate(pad - 2 * dpr, H - pad);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'left';
      ctx.fillText('Δy (km)', 0, 0);
      ctx.restore();

      // legend for rings (with probability mass)
      const legendItems = rings.map(r => {
        const p = 1 - Math.exp(-0.5 * r.k * r.k);
        return { label: `${r.k}σ (${(100 * p).toFixed(1)}%)`, color: r.color };
      });
      this._drawLegend(ctx, pad, dpr, legendItems);

      // scale badge
      const spanKm = this._halfRange / Math.max(1e-9, this.zoom);
      const badge = `±${spanKm.toFixed(spanKm < 1 ? 2 : 1)} km`;
      const bx = W - pad;
      const by = pad;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      const tw = (badge.length * 7.2) * dpr;
      ctx.fillRect(bx - tw - 10 * dpr, by, tw + 10 * dpr, 18 * dpr);
      ctx.fillStyle = 'rgba(255,255,255,0.82)';
      ctx.fillText(badge, bx - 4 * dpr, by + 2 * dpr);

      ctx.restore();
    }
  }

  // scratch storage for trail points (px,py)
  RelPosCovPlot._trailPtsCap = 900; // must be >= _trailMaxPts margin
  RelPosCovPlot._trailPts = new Float32Array(RelPosCovPlot._trailPtsCap * 2);

  class OrbitView {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d', { alpha: true });
      this.dpr = 1;
      this.w = 0;
      this.h = 0;

      this.centerX = 0;
      this.centerY = 0;
      this.scale = 0.06; // px per km

      this.followCraft = false;
      this.drag = false;
      this.dragStart = { x: 0, y: 0, cx: 0, cy: 0 };
      this._bind();
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (w !== this.canvas.width || h !== this.canvas.height) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.dpr = dpr;
      this.w = w;
      this.h = h;
    }

    _bind() {
      const c = this.canvas;

      c.addEventListener('mousedown', (e) => {
        this.drag = true;
        this.followCraft = false;
        const r = c.getBoundingClientRect();
        this.dragStart.x = (e.clientX - r.left) * (window.devicePixelRatio || 1);
        this.dragStart.y = (e.clientY - r.top) * (window.devicePixelRatio || 1);
        this.dragStart.cx = this.centerX;
        this.dragStart.cy = this.centerY;
        app.ui.btnZoomCraft.textContent = 'Zoom to Craft';
      });

      window.addEventListener('mouseup', () => { this.drag = false; });

      window.addEventListener('mousemove', (e) => {
        if (!this.drag) return;
        const r = c.getBoundingClientRect();
        const mx = (e.clientX - r.left) * (window.devicePixelRatio || 1);
        const my = (e.clientY - r.top) * (window.devicePixelRatio || 1);
        const dx = mx - this.dragStart.x;
        const dy = my - this.dragStart.y;
        this.centerX = this.dragStart.cx - dx / Math.max(EPS, this.scale);
        this.centerY = this.dragStart.cy + dy / Math.max(EPS, this.scale);
      });

      c.addEventListener('wheel', (e) => {
        e.preventDefault();
        const factor = (e.deltaY < 0) ? 1.12 : 0.89;
        const old = this.scale;
        const ns = clamp(old * factor, 0.0005, 1.2);

        const rect = c.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (window.devicePixelRatio || 1);
        const my = (e.clientY - rect.top) * (window.devicePixelRatio || 1);
        const wx = this.screenToWorldX(mx);
        const wy = this.screenToWorldY(my);

        this.scale = ns;

        const wx2 = this.screenToWorldX(mx);
        const wy2 = this.screenToWorldY(my);
        this.centerX += (wx - wx2);
        this.centerY += (wy - wy2);
      }, { passive: false });
    }

    worldToScreenX(x) { return (x - this.centerX) * this.scale + this.w * 0.5; }
    worldToScreenY(y) { return (-(y - this.centerY)) * this.scale + this.h * 0.5; }
    screenToWorldX(px) { return (px - this.w * 0.5) / Math.max(EPS, this.scale) + this.centerX; }
    screenToWorldY(py) { return (-(py - this.h * 0.5)) / Math.max(EPS, this.scale) + this.centerY; }

    resetView() {
      this.resize();
      this.centerX = 0;
      this.centerY = 0;

      let maxR = 0;
      const buf = app.hist && app.hist.truth ? app.hist.truth : null;
      if (buf && buf.len > 0) {
        buf.forEachChronological((_t, data, off) => {
          const x = data[off + 0], y = data[off + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y)) return;
          const r = Math.hypot(x, y);
          if (r > maxR) maxR = r;
        });
      }

      if (!(maxR > 0)) {
        const rp = app.state.rp || (R_E + 300);
        const e = clamp(app.state.ecc, 0, 0.6);
        const a = rp / Math.max(EPS, (1 - e));
        maxR = a * (1 + e);
      }

      const pad = 1.20;
      const extent = Math.max(R_E * 1.15, maxR * pad);
      const s = Math.min(this.w, this.h) / Math.max(EPS, 2 * extent);
      this.scale = clamp(s, 0.0005, 1.2);

      this.followCraft = false;
      app.ui.btnZoomCraft.textContent = 'Zoom to Craft';
    }

    zoomToCraft() {
      this.resize();
      this.followCraft = true;
      app.ui.btnZoomCraft.textContent = 'Following EKF';
      const halfSpanKm = 3000;
      const targetScale = Math.min(this.w, this.h) / Math.max(EPS, 2 * halfSpanKm);
      this.scale = clamp(Math.max(this.scale, targetScale), 0.0005, 1.2);
    }

    draw() {
      this.resize();
      const ctx = this.ctx;
      const W = this.w, H = this.h;
      const xTrue = app.state.xTrue;
      const xHat = app.state.xHat;

      if (this.followCraft) {
        this.centerX = xHat[0];
        this.centerY = xHat[2];
      }

      ctx.save();
      ctx.clearRect(0, 0, W, H);

      const bg = ctx.createRadialGradient(W * 0.5, H * 0.35, 10, W * 0.5, H * 0.5, Math.max(W, H));
      bg.addColorStop(0, 'rgba(255,255,255,0.04)');
      bg.addColorStop(1, 'rgba(0,0,0,0.28)');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      for (let i = 0; i < 120; i++) {
        const x = (i * 1103 % W);
        const y = (i * 577 % H);
        ctx.fillRect(x, y, 1, 1);
      }

      // Earth
      const ex = this.worldToScreenX(0);
      const ey = this.worldToScreenY(0);
      const eR = Math.max(0.5, R_E * this.scale);
      const eRsafe = Math.max(0.5, eR);

      const grad = ctx.createRadialGradient(
        ex - 0.25 * eRsafe, ey - 0.30 * eRsafe, 0.15 * eRsafe,
        ex, ey, eRsafe * 1.15
      );
      grad.addColorStop(0, 'rgba(220,240,255,0.95)');
      grad.addColorStop(0.35, 'rgba(80,150,255,0.75)');
      grad.addColorStop(0.7, 'rgba(25,55,120,0.80)');
      grad.addColorStop(1, 'rgba(0,0,0,0.15)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(ex, ey, eRsafe, 0, TWO_PI);
      ctx.fill();

      ctx.strokeStyle = 'rgba(255,255,255,0.14)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(ex, ey, eRsafe, 0, TWO_PI);
      ctx.stroke();

      // Stations
      const t = app.state.t;
      const pv = OrbitView._pv;
      for (let i = 0; i < 12; i++) {
        stationState(i, t, pv);
        const sx = this.worldToScreenX(pv[0]);
        const sy = this.worldToScreenY(pv[2]);
        const c = app.stationColors[i];

        ctx.fillStyle = c;
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(2.2, 3.0 * (this.dpr)), 0, TWO_PI);
        ctx.fill();

        const isVis = app.state.visMask[i] === 1;
        const tx = this.worldToScreenX(xTrue[0]);
        const ty = this.worldToScreenY(xTrue[2]);
        let ang = Math.atan2(ty - sy, tx - sx);
        if (!isVis) ang = Math.atan2(sy - ey, sx - ex);

        const rr = Math.max(6, 8 * this.dpr);
        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(ang);
        ctx.fillStyle = isVis ? c : 'rgba(255,255,255,0.12)';
        ctx.beginPath();
        ctx.moveTo(rr, 0);
        ctx.lineTo(-rr * 0.6, rr * 0.45);
        ctx.lineTo(-rr * 0.6, -rr * 0.45);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      // Signal links + measurement dots (aligned to measurement update)
      const liveT = getLiveSimTime();
      const lastMeasT = app.state.lastMeasT;
      const dtStep = Math.max(1e-6, app.state.dt);
      const tau = liveT - lastMeasT;
      const phase = clamp(tau / dtStep, 0, 1);

      const txs = this.worldToScreenX(xTrue[0]);
      const tys = this.worldToScreenY(xTrue[2]);

      for (let i = 0; i < 12; i++) {
        if (app.state.visMask[i] !== 1) continue;
        stationState(i, t, pv);
        const sx = this.worldToScreenX(pv[0]);
        const sy = this.worldToScreenY(pv[2]);
        const c = app.stationColors[i];

        ctx.strokeStyle = c;
        ctx.globalAlpha = 0.20;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(txs, tys);
        ctx.stroke();

        ctx.globalAlpha = 0.55;
        ctx.setLineDash([8 * this.dpr, 10 * this.dpr]);
        ctx.lineDashOffset = -phase * 30 * this.dpr;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(txs, tys);
        ctx.stroke();
        ctx.setLineDash([]);

        // Dot only during the interval after a measurement update
        if (tau >= 0 && tau <= dtStep) {
          const p = phase;
          const mx = txs * (1 - p) + sx * p;
          const my = tys * (1 - p) + sy * p;
          ctx.globalAlpha = 0.95;
          ctx.fillStyle = c;
          ctx.beginPath();
          ctx.arc(mx, my, Math.max(2.0, 2.6 * this.dpr), 0, TWO_PI);
          ctx.fill();
        }
      }

      ctx.globalAlpha = 1;

      const drawPath = (buf, color, alpha) => {
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        buf.forEachChronological((_tt, data, off) => {
          const x = data[off + 0], y = data[off + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y)) return;
          const px = this.worldToScreenX(x);
          const py = this.worldToScreenY(y);
          if (!started) { ctx.moveTo(px, py); started = true; }
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      };

      drawPath(app.hist.truth, 'rgba(27,110,234,0.95)', 0.55);
      drawPath(app.hist.est, 'rgba(255,122,24,0.95)', 0.45);

      // Craft markers
      const tx = this.worldToScreenX(xTrue[0]);
      const ty = this.worldToScreenY(xTrue[2]);
      const hx = this.worldToScreenX(xHat[0]);
      const hy = this.worldToScreenY(xHat[2]);

      ctx.save();
      ctx.shadowColor = 'rgba(27,110,234,0.55)';
      ctx.shadowBlur = 10 * this.dpr;
      ctx.fillStyle = 'rgba(27,110,234,0.95)';
      ctx.beginPath();
      ctx.arc(tx, ty, Math.max(3.4, 4.4 * this.dpr), 0, TWO_PI);
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = 'rgba(255,122,24,0.95)';
      ctx.beginPath();
      ctx.rect(hx - 3.0 * this.dpr, hy - 3.0 * this.dpr, 6.0 * this.dpr, 6.0 * this.dpr);
      ctx.fill();

      this._drawThrustVector(hx, hy);
      this._drawCovEllipse(xHat, app.state.P);

      ctx.restore();
    }

    _drawThrustVector(px, py) {
      const ctx = this.ctx;
      const u = app.state.uEci;
      const umax = Math.max(EPS, app.state.umax);
      const ux = u[0], uy = u[1];
      const mag = Math.hypot(ux, uy);

      // Required: disappear when no forcing is occurring
      if (!Number.isFinite(mag) || mag <= 1e-18) return;

      const frac = clamp(mag / umax, 0, 1);
      const len = (10 + 44 * frac) * (this.dpr);
      const vx = ux / Math.max(EPS, mag);
      const vy = uy / Math.max(EPS, mag);
      const dx = (vx) * len;
      const dy = (-vy) * len;

      ctx.save();
      ctx.translate(px, py);
      ctx.strokeStyle = 'rgba(124,58,237,0.92)';
      ctx.lineWidth = 2.4 * this.dpr;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(dx, dy);
      ctx.stroke();

      const ang = Math.atan2(dy, dx);
      const ah = 8 * this.dpr;
      ctx.fillStyle = 'rgba(124,58,237,0.92)';
      ctx.beginPath();
      ctx.moveTo(dx, dy);
      ctx.lineTo(dx - ah * Math.cos(ang - Math.PI / 7), dy - ah * Math.sin(ang - Math.PI / 7));
      ctx.lineTo(dx - ah * Math.cos(ang + Math.PI / 7), dy - ah * Math.sin(ang + Math.PI / 7));
      ctx.closePath();
      ctx.fill();

      if (app.state.saturated) {
        ctx.strokeStyle = 'rgba(220,38,38,0.55)';
        ctx.lineWidth = 1.4 * this.dpr;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(dx, dy);
        ctx.stroke();
      }
      ctx.restore();
    }

    _drawCovEllipse(xHat, P) {
      const ctx = this.ctx;

      // (X,Y) cov is indices [0,2; 8,10] in the 4x4 flattened row-major
      const px = P[0], py = P[10], pxy = P[2];

      const tr = px + py;
      const det = px * py - pxy * pxy;
      const disc = Math.max(0, tr * tr * 0.25 - det);
      const s = Math.sqrt(disc);

      let l1 = tr * 0.5 + s;
      let l2 = tr * 0.5 - s;
      l1 = Math.max(1e-16, l1);
      l2 = Math.max(1e-16, l2);

      const ang = 0.5 * Math.atan2(2 * pxy, px - py);
      const k = 2.0; // 2-sigma

      const a = k * Math.sqrt(l1);
      const b = k * Math.sqrt(l2);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;

      const ax = Math.max(0.5, a * this.scale);
      const bx = Math.max(0.5, b * this.scale);

      const cx = this.worldToScreenX(xHat[0]);
      const cy = this.worldToScreenY(xHat[2]);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-ang);
      ctx.strokeStyle = 'rgba(255,122,24,0.45)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(0, 0, ax, bx, 0, 0, TWO_PI);
      ctx.stroke();
      ctx.restore();
    }
  }
  OrbitView._pv = new Float64Array(4);

  // ---------- Application state ----------
  const app = {
    state: {
      running: false,
      speed: 10, // default 10x
      dt: 10,
      t: 0,
      step: 0,

      rp: R_E + 300,
      ecc: 0,

      xTrue: new Float64Array(4),
      xHat: new Float64Array(4),
      P: new Float64Array(16),

      ctrlMode: 'off',
      umax: 0.01 * G0,
      ref: {
        r0: R_E + 300,
        n0: Math.sqrt(MU / Math.pow(R_E + 300, 3)),
        theta0: 0
      },
      lastUrt: new Float64Array(4), // [ur, ut, |u|, sat]
      uEci: new Float64Array(2),

      measNoise: true,

      // disturbance: constant acceleration bias applied to truth
      disturb: false,
      sigA: 1e-7,
      aBias: new Float64Array(2),

      sigRho: 0.1,
      sigRhoDot: 1.0,
      sigPhi: 0.1,

      visMask: new Uint8Array(12),

      nees: NaN,
      nis: NaN,
      neesLo: NaN,
      neesHi: NaN,
      nisLo: NaN,
      nisHi: NaN,

      innovNorm: 0,
      mVis: 0,

      impacted: false,
      saturated: false,

      plotSpanSec: 10800, // default 3h
      activeTab: 'tabStates',

      lastMeasT: 0,
      lastPerturbT: -1e9
    },
    hist: {
      truth: null,
      est: null,
      ctrl: null,
      neesnis: null,
      sigmas: null,
      errComp: null
    },
    measHist: { perStation: [] },
    ui: {},
    plots: {},
    orbitView: null,
    stationColors: [],
    rng: null,
    animFrame: 0,
    clock: { lastTs: 0, accum: 0 },
    stationTiles: [],
    relPosPlot: null
  };

  function initColors() {
    const cols = [];
    for (let i = 0; i < 12; i++) {
      const hue = (i * 29 + 10) % 360;
      cols.push(`hsla(${hue}, 85%, 65%, 0.95)`);
    }
    app.stationColors = cols;
  }

  function getLiveSimTime() {
    const s = app.state;
    if (s.running) return s.t + app.clock.accum;
    return s.t;
  }

  function computeNEES(xTrue, xHat, P) {
    const e0 = xTrue[0] - xHat[0];
    const e1 = xTrue[1] - xHat[1];
    const e2 = xTrue[2] - xHat[2];
    const e3 = xTrue[3] - xHat[3];

    const Pinv = computeNEES._Pinv;
    const ok = matInv4(P, Pinv);
    if (!ok) return NaN;

    const z0 = Pinv[0] * e0 + Pinv[1] * e1 + Pinv[2] * e2 + Pinv[3] * e3;
    const z1 = Pinv[4] * e0 + Pinv[5] * e1 + Pinv[6] * e2 + Pinv[7] * e3;
    const z2 = Pinv[8] * e0 + Pinv[9] * e1 + Pinv[10] * e2 + Pinv[11] * e3;
    const z3 = Pinv[12] * e0 + Pinv[13] * e1 + Pinv[14] * e2 + Pinv[15] * e3;

    return e0 * z0 + e1 * z1 + e2 * z2 + e3 * z3;
  }
  computeNEES._Pinv = new Float64Array(16);

  function updateCtrlSatPanel() {
    const s = app.state;
    if (!app.ui.ctrlSatBox) return;

    const umax = Math.max(EPS, s.umax);
    const umag = Math.max(0, safeNum(s.lastUrt[2], 0));
    const frac = umag / umax;

    app.ui.ctrlSatFrac.textContent = Number.isFinite(frac) ? frac.toFixed(2) : '–';

    const sat = !!s.saturated;
    app.ui.ctrlSatText.textContent = sat ? 'SATURATED' : 'OK';
    app.ui.ctrlSatBox.classList.toggle('bad', sat);
    app.ui.ctrlSatBox.classList.toggle('ok', !sat);
  }

  function updateWarnPills() {
    const s = app.state;

    const pillDisturb = app.ui.pillDisturb;
    pillDisturb.textContent = s.disturb ? 'Disturbance ON' : 'Disturbance OFF';
    pillDisturb.style.background = s.disturb ? 'rgba(220,38,38,0.20)' : 'rgba(180,83,9,0.18)';
    pillDisturb.style.borderColor = s.disturb ? 'rgba(220,38,38,0.34)' : 'rgba(180,83,9,0.25)';

    const pillSat = app.ui.pillSat;
    pillSat.textContent = s.saturated ? 'SATURATING' : 'Saturation OK';
    pillSat.style.background = s.saturated ? 'rgba(220,38,38,0.20)' : 'rgba(180,83,9,0.18)';
    pillSat.style.borderColor = s.saturated ? 'rgba(220,38,38,0.34)' : 'rgba(180,83,9,0.25)';

    app.ui.pillImpact.style.display = s.impacted ? 'inline-flex' : 'none';

    updateCtrlSatPanel();
  }

  function updateHUD() {
    const s = app.state;
    const liveT = getLiveSimTime();
    app.ui.hudTime.textContent = liveT.toFixed(2);
    app.ui.hudStep.textContent = `${s.step}`;
    app.ui.hudDt.textContent = `${s.dt}`;
    app.ui.hudVisible.textContent = `${s.mVis}`;
    app.ui.hudMode.textContent = s.impacted ? 'Impact (Paused)' : (s.running ? 'Running' : 'Paused');
    app.ui.hudSpeed.textContent = `${s.speed}×`;
    app.ui.hudNees.textContent = Number.isFinite(s.nees) ? s.nees.toFixed(2) : '–';
    app.ui.hudNis.textContent = Number.isFinite(s.nis) ? s.nis.toFixed(2) : '–';
  }

  function setSpeed(val) {
    const s = app.state;
    const v = clamp(parseInt(val, 10) || 10, 1, 10000);
    s.speed = v;
    for (const b of app.ui.speedButtons) {
      b.classList.toggle('active', parseInt(b.dataset.speed, 10) === v);
    }
    updateHUD();
  }

  function controllerAllowed() {
    return Math.abs(app.state.ecc) < 1e-12;
  }

  function enforceControllerAvailability() {
    const s = app.state;
    const allowed = controllerAllowed();
    if (!allowed) {
      // force off and disable UI
      s.ctrlMode = 'off';
      app.ui.selCtrl.value = 'off';
      app.ui.selCtrl.disabled = true;
      app.ui.inpUmaxG.disabled = true;
      app.ui.ctrlEccNote.style.display = 'block';
    } else {
      app.ui.selCtrl.disabled = false;
      app.ui.inpUmaxG.disabled = false;
      app.ui.ctrlEccNote.style.display = 'none';
    }
  }

  function syncUiToState() {
    const s = app.state;

    const dt = clamp(safeNum(parseFloat(app.ui.inpDt.value), s.dt), 1, 120);
    s.dt = dt;
    if (app.ui.inpDt.value !== `${dt}`) app.ui.inpDt.value = `${dt}`;

    s.measNoise = (app.ui.selMeasNoise.value === 'on');

    const ecc = clamp(safeNum(parseFloat(app.ui.inpEcc.value), s.ecc), 0, 0.6);
    s.ecc = ecc;
    if (app.ui.inpEcc.value !== `${ecc}`) app.ui.inpEcc.value = `${ecc}`;

    // disturbance toggle
    const newDisturb = (app.ui.selDisturb.value === 'on');
    if (newDisturb !== s.disturb) {
      s.disturb = newDisturb;
      if (s.disturb) {
        // pick a fixed direction for the bias (constant accel)
        const ang = TWO_PI * app.rng.nextFloat();
        s.aBias[0] = Math.cos(ang);
        s.aBias[1] = Math.sin(ang);
      } else {
        s.aBias[0] = 0;
        s.aBias[1] = 0;
      }
      updateWarnPills();
    }

    s.sigRho = Math.max(0, safeNum(parseFloat(app.ui.inpSigRho.value), s.sigRho));
    s.sigRhoDot = Math.max(0, safeNum(parseFloat(app.ui.inpSigRhoDot.value), s.sigRhoDot));
    s.sigPhi = Math.max(0, safeNum(parseFloat(app.ui.inpSigPhi.value), s.sigPhi));
    s.sigA = Math.max(0, safeNum(parseFloat(app.ui.inpSigA.value), s.sigA));

    // controller selection (only off/lqr)
    enforceControllerAvailability();
    s.ctrlMode = app.ui.selCtrl.value;

    const umaxG = Math.max(0, safeNum(parseFloat(app.ui.inpUmaxG.value), s.umax / G0));
    s.umax = umaxG * G0;

    const span = clamp(safeNum(parseFloat(app.ui.selPlotWindow.value), s.plotSpanSec), 600, 86400);
    s.plotSpanSec = span;
  }

  function createStationBoard() {
    const board = app.ui.stationBoard;
    if (!board) return;

    board.innerHTML = '';
    app.stationTiles = [];

    for (let i = 0; i < 12; i++) {
      const tile = document.createElement('div');
      tile.className = 'station-tile';
      tile.dataset.sid = `${i}`;
      tile.style.setProperty('--sc', app.stationColors[i]);

      const icon = document.createElement('div');
      icon.className = 'station-icon';
      icon.innerHTML = '<i data-lucide="satellite-dish"></i>';

      const sig = document.createElement('div');
      sig.className = 'station-signal';

      const id = document.createElement('div');
      id.className = 'station-id';
      id.textContent = `ST-${String(i + 1).padStart(2, '0')}`;

      const state = document.createElement('div');
      state.className = 'station-state';
      state.textContent = '-';

      tile.appendChild(icon);
      tile.appendChild(sig);
      tile.appendChild(id);
      tile.appendChild(state);

      board.appendChild(tile);
      app.stationTiles.push({ el: tile, state });
    }

    // render lucide icons
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function updateStationBoard() {
    for (let i = 0; i < app.stationTiles.length; i++) {
      const t = app.stationTiles[i];
      const active = app.state.visMask[i] === 1;
      t.el.classList.toggle('active', active);
      t.state.textContent = active ? 'ACTIVE' : '-';
    }
  }

  function resetSim() {
    const s = app.state;
    s.t = 0;
    s.step = 0;
    s.impacted = false;
    s.saturated = false;
    s.lastMeasT = 0;
    s.lastPerturbT = -1e9;

    const perAlt = clamp(safeNum(parseFloat(app.ui.inpPerigeeAlt.value), 300), 150, 2000);
    app.ui.inpPerigeeAlt.value = `${perAlt}`;

    const ecc = clamp(safeNum(parseFloat(app.ui.inpEcc.value), 0), 0, 0.6);
    app.ui.inpEcc.value = `${ecc}`;
    s.ecc = ecc;

    const rp = R_E + perAlt;
    s.rp = rp;
    s.ref.r0 = rp;
    s.ref.n0 = Math.sqrt(MU / Math.pow(rp, 3));

    // initialize truth at perigee on +X axis
    const vPer = Math.sqrt(MU * (1 + ecc) / Math.max(EPS, rp));
    s.xTrue[0] = rp;  s.xTrue[1] = 0;
    s.xTrue[2] = 0;   s.xTrue[3] = vPer;

    // initial estimate
    s.xHat[0] = rp + 0.3;
    s.xHat[1] = 0.0002;
    s.xHat[2] = 0.2;
    s.xHat[3] = vPer - 0.0002;

    s.ref.theta0 = Math.atan2(s.xTrue[2], s.xTrue[0]);

    mat4Identity(s.P);
    s.P[0] = 0.5 * 0.5;
    s.P[5] = 1e-4;
    s.P[10] = 0.5 * 0.5;
    s.P[15] = 1e-4;

    s.lastUrt[0] = 0; s.lastUrt[1] = 0; s.lastUrt[2] = 0; s.lastUrt[3] = 0;
    s.uEci[0] = 0; s.uEci[1] = 0;

    // disturbance bias direction
    s.aBias[0] = 0;
    s.aBias[1] = 0;

    for (let i = 0; i < 12; i++) s.visMask[i] = 0;

    const CAP_MAIN = 45000;
    const CAP_MEAS = 20000;

    app.hist.truth = new RingBuffer(CAP_MAIN, 4);
    app.hist.est = new RingBuffer(CAP_MAIN, 4);
    app.hist.ctrl = new RingBuffer(CAP_MAIN, 4);      // [ur, ut, |u|, sat]
    app.hist.neesnis = new RingBuffer(CAP_MAIN, 6);   // [nees, lo, hi, nis, lo, hi]
    app.hist.sigmas = new RingBuffer(CAP_MAIN, 4);    // [sx, sy, svx, svy]
    app.hist.errComp = new RingBuffer(CAP_MAIN, 4);   // [ex, ey, evx, evy]

    app.measHist.perStation = [];
    for (let i = 0; i < 12; i++) app.measHist.perStation.push(new RingBuffer(CAP_MEAS, 6));

    app.orbitView.resetView();

    app.hist.truth.push(0, s.xTrue);
    app.hist.est.push(0, s.xHat);
    app.hist.ctrl.push(0, new Float64Array([0, 0, 0, 0]));
    app.hist.neesnis.push(0, new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN]));
    app.hist.sigmas.push(0, new Float64Array([
      Math.sqrt(s.P[0]),
      Math.sqrt(s.P[10]),
      Math.sqrt(s.P[5]),
      Math.sqrt(s.P[15])
    ]));
    app.hist.errComp.push(0, new Float64Array([
      s.xTrue[0] - s.xHat[0],
      s.xTrue[2] - s.xHat[2],
      s.xTrue[1] - s.xHat[1],
      s.xTrue[3] - s.xHat[3]
    ]));

    app.clock.lastTs = 0;
    app.clock.accum = 0;

    enforceControllerAvailability();
    updateWarnPills();
    updateStationBoard();
    updateHUD();
    updateCtrlNotice();
  }

  function perturbOrbit() {
    const s = app.state;
    // small, visible velocity kick (about 1 m/s)
    const dv = 0.001; // km/s
    const ang = TWO_PI * app.rng.nextFloat();
    s.xTrue[1] += dv * Math.cos(ang);
    s.xTrue[3] += dv * Math.sin(ang);
    s.lastPerturbT = s.t;
  }

  // ---------- Simulation step ----------
  function stepOnce() {
    const s = app.state;
    if (s.impacted) return;

    syncUiToState();

    const dt = s.dt;
    const t = s.t;

    // --- Controller uses ESTIMATE ---
    const xCtrl = stepOnce._xCtrl;
    let ur = 0, ut = 0;

    if (s.ctrlMode === 'lqr' && controllerAllowed()) {
      const pol = eciToPolarDeviations(s.xHat, t, s.ref);
      xCtrl[0] = pol.dr;
      xCtrl[1] = pol.drdot;
      xCtrl[2] = pol.dtheta;
      xCtrl[3] = pol.dthetadot;

      const urut = stepOnce._urut;
      controlLawLqr(xCtrl, urut);

      const sat = stepOnce._sat;
      saturateU(urut[0], urut[1], s.umax, sat);

      ur = sat[0];
      ut = sat[1];

      s.lastUrt[0] = sat[0];
      s.lastUrt[1] = sat[1];
      s.lastUrt[2] = sat[2];
      s.lastUrt[3] = sat[3];

      s.saturated = (sat[3] === 1);
      radialTangentialToEci(pol.theta, ur, ut, s.uEci);
    } else {
      s.uEci[0] = 0;
      s.uEci[1] = 0;
      s.lastUrt[0] = 0;
      s.lastUrt[1] = 0;
      s.lastUrt[2] = 0;
      s.lastUrt[3] = 0;
      s.saturated = false;

      xCtrl[0] = 0;
      xCtrl[1] = 0;
      xCtrl[2] = 0;
      xCtrl[3] = 0;
    }

    // --- Truth disturbance accel (constant bias) ---
    const aDist = stepOnce._aDist;
    if (s.disturb) {
      const mag = s.sigA;
      aDist[0] = mag * s.aBias[0];
      aDist[1] = mag * s.aBias[1];
    } else {
      aDist[0] = 0;
      aDist[1] = 0;
    }

    // --- Truth propagation ---
    const xNext = stepOnce._xNext;
    rk4Step(s.xTrue, s.uEci, aDist, dt, xNext);
    for (let i = 0; i < 4; i++) s.xTrue[i] = xNext[i];

    const r = Math.sqrt(Math.max(EPS, s.xTrue[0] * s.xTrue[0] + s.xTrue[2] * s.xTrue[2]));
    if (r < R_E) {
      s.impacted = true;
      s.running = false;
      app.ui.btnRunPause.textContent = 'Run';
      updateWarnPills();
      updateCtrlNotice();
      return;
    }

    // --- Measurements ---
    const visIds = stepOnce._visIds;
    const mVis = visibleStations(s.xTrue, t + dt, visIds);
    s.mVis = mVis;

    for (let i = 0; i < 12; i++) s.visMask[i] = 0;
    for (let k = 0; k < mVis; k++) s.visMask[visIds[k]] = 1;

    const yTrue = stepOnce._yTrue;
    const Htmp = stepOnce._Htmp;
    H_jacobian(s.xTrue, t + dt, visIds, mVis, yTrue, Htmp);

    if (s.measNoise) {
      for (let k = 0; k < mVis; k++) {
        const idx = 3 * k;
        yTrue[idx + 0] += s.sigRho * randn(app.rng.nextFloat);
        yTrue[idx + 1] += s.sigRhoDot * randn(app.rng.nextFloat);
        yTrue[idx + 2] = wrapPi(yTrue[idx + 2] + s.sigPhi * randn(app.rng.nextFloat));
      }
    }

    // --- EKF propagation ---
    const Q = stepOnce._Q;
    Q[0] = 1e-10; Q[1] = 0;
    Q[2] = 0;     Q[3] = 1e-10;
    ekfPredict(s.xHat, s.P, s.uEci, Q, dt);

    // --- EKF update ---
    const Rper = stepOnce._Rper;
    Rper[0] = s.sigRho * s.sigRho;     Rper[1] = 0; Rper[2] = 0;
    Rper[3] = 0;                       Rper[4] = s.sigRhoDot * s.sigRhoDot; Rper[5] = 0;
    Rper[6] = 0;                       Rper[7] = 0; Rper[8] = s.sigPhi * s.sigPhi;

    const diag = stepOnce._diag;
    ekfUpdate(s.xHat, s.P, yTrue, t + dt, visIds, mVis, Rper, diag);

    s.innovNorm = diag.innovNorm;
    s.nis = diag.nis;
    s.nees = computeNEES(s.xTrue, s.xHat, s.P);

    const alpha = 0.05;
    s.neesLo = chi2Inv(alpha / 2, 4);
    s.neesHi = chi2Inv(1 - alpha / 2, 4);

    const dfNis = Math.max(1, 3 * mVis);
    s.nisLo = chi2Inv(alpha / 2, dfNis);
    s.nisHi = chi2Inv(1 - alpha / 2, dfNis);

    // advance time
    s.t = t + dt;
    s.step++;
    s.lastMeasT = s.t; // for aligned orbit-view dots

    // histories
    app.hist.truth.push(s.t, s.xTrue);
    app.hist.est.push(s.t, s.xHat);
    app.hist.ctrl.push(s.t, new Float64Array([s.lastUrt[0], s.lastUrt[1], s.lastUrt[2], s.lastUrt[3]]));
    app.hist.neesnis.push(s.t, new Float64Array([s.nees, s.neesLo, s.neesHi, s.nis, s.nisLo, s.nisHi]));

    const sx = Math.sqrt(Math.max(0, s.P[0]));
    const sy = Math.sqrt(Math.max(0, s.P[10]));
    const svx = Math.sqrt(Math.max(0, s.P[5]));
    const svy = Math.sqrt(Math.max(0, s.P[15]));
    app.hist.sigmas.push(s.t, new Float64Array([sx, sy, svx, svy]));

    app.hist.errComp.push(s.t, new Float64Array([
      s.xTrue[0] - s.xHat[0],
      s.xTrue[2] - s.xHat[2],
      s.xTrue[1] - s.xHat[1],
      s.xTrue[3] - s.xHat[3]
    ]));

    // measurement histories per station
    const yHat = stepOnce._yHat;
    const Hhat = stepOnce._Hhat;
    H_jacobian(s.xHat, s.t, visIds, mVis, yHat, Hhat);

    for (let k = 0; k < mVis; k++) {
      const sid = visIds[k];
      const idx = 3 * k;
      app.measHist.perStation[sid].push(s.t, new Float64Array([
        yTrue[idx + 0], yHat[idx + 0],
        yTrue[idx + 1], yHat[idx + 1],
        yTrue[idx + 2], yHat[idx + 2],
      ]));
    }

    updateWarnPills();
    updateStationBoard();
    updateCtrlNotice();
  }

  stepOnce._xCtrl = new Float64Array(4);
  stepOnce._urut = new Float64Array(2);
  stepOnce._sat = new Float64Array(4);
  stepOnce._aDist = new Float64Array(2);
  stepOnce._xNext = new Float64Array(4);
  stepOnce._visIds = new Int16Array(12);
  stepOnce._yTrue = new Float64Array(36);
  stepOnce._Htmp = new Float64Array(36 * 4);
  stepOnce._Q = new Float64Array(4);
  stepOnce._Rper = new Float64Array(9);
  stepOnce._diag = { innovNorm: 0, nis: NaN, sSingular: false };
  stepOnce._yHat = new Float64Array(36);
  stepOnce._Hhat = new Float64Array(36 * 4);

  // ---------- Plot setup ----------
  function setupPlots() {
    const spanFn = () => app.state.plotSpanSec;

    app.plots.pltX = new PlotCanvas(app.ui.pltX, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'X (km)' });
    app.plots.pltY = new PlotCanvas(app.ui.pltY, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'Y (km)' });
    app.plots.pltXd = new PlotCanvas(app.ui.pltXd, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'Ẋ (km/s)' });
    app.plots.pltYd = new PlotCanvas(app.ui.pltYd, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'Ẏ (km/s)' });

    app.plots.pltErrX = new PlotCanvas(app.ui.pltErrX, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'eX (km)' });
    app.plots.pltErrY = new PlotCanvas(app.ui.pltErrY, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'eY (km)' });
    app.plots.pltErrXd = new PlotCanvas(app.ui.pltErrXd, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'eẊ (km/s)' });
    app.plots.pltErrYd = new PlotCanvas(app.ui.pltErrYd, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'eẎ (km/s)' });

    app.plots.pltRho = new PlotCanvas(app.ui.pltRho, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'ρ (km)' });
    app.plots.pltRhoDot = new PlotCanvas(app.ui.pltRhoDot, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'ρ̇ (km/s)' });
    app.plots.pltPhi = new PlotCanvas(app.ui.pltPhi, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'φ (rad)' });

    app.plots.pltNEES = new PlotCanvas(app.ui.pltNEES, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'NEES (—)' });
    app.plots.pltNIS = new PlotCanvas(app.ui.pltNIS, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'NIS (—)' });

    app.plots.pltUrt = new PlotCanvas(app.ui.pltUrt, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'u (km/s²)' });
    app.plots.pltUm = new PlotCanvas(app.ui.pltUm, {
      xSpanFn: spanFn,
      xLabel: 't (s)',
      yLabel: '‖u‖ (km/s²)',
      fixedYFn: () => [0, Math.max(app.state.umax * 1.25, 1e-6)]
    });

    app.plots.pltCtrlErr = new PlotCanvas(app.ui.pltCtrlErr, { xSpanFn: spanFn, xLabel: 't (s)', yLabel: 'ctrl (—)', minYSpan: 2.0 });
  }

  function drawPlots() {
    const s = app.state;
    const truth = app.hist.truth;
    const est = app.hist.est;
    const ctrl = app.hist.ctrl;
    const nn = app.hist.neesnis;
    const sig = app.hist.sigmas;
    const err = app.hist.errComp;

    if (s.activeTab === 'tabStates') {
      app.plots.pltX.draw([
        makeLineSeries(truth, 0, 'rgba(27,110,234,0.95)', 'truth'),
        makeLineSeries(est, 0, 'rgba(255,122,24,0.95)', 'ekf'),
      ]);
      app.plots.pltY.draw([
        makeLineSeries(truth, 2, 'rgba(27,110,234,0.95)', 'truth'),
        makeLineSeries(est, 2, 'rgba(255,122,24,0.95)', 'ekf'),
      ]);
      app.plots.pltXd.draw([
        makeLineSeries(truth, 1, 'rgba(27,110,234,0.95)', 'truth'),
        makeLineSeries(est, 1, 'rgba(255,122,24,0.95)', 'ekf'),
      ]);
      app.plots.pltYd.draw([
        makeLineSeries(truth, 3, 'rgba(27,110,234,0.95)', 'truth'),
        makeLineSeries(est, 3, 'rgba(255,122,24,0.95)', 'ekf'),
      ]);
      return;
    }

    if (s.activeTab === 'tabErrors') {
      const errSeries = (errDim, sigDim, color) => ([
        makeLineSeries(err, errDim, color, 'error'),
        makeScaledLineSeries(sig, sigDim, 2.0, 'rgba(255,255,255,0.55)', '+2σ'),
        makeScaledLineSeries(sig, sigDim, -2.0, 'rgba(255,255,255,0.55)', '−2σ'),
      ]);

      app.plots.pltErrX.draw(errSeries(0, 0, 'rgba(255,122,24,0.95)'));
      app.plots.pltErrY.draw(errSeries(1, 1, 'rgba(255,122,24,0.95)'));
      app.plots.pltErrXd.draw(errSeries(2, 2, 'rgba(255,122,24,0.95)'));
      app.plots.pltErrYd.draw(errSeries(3, 3, 'rgba(255,122,24,0.95)'));
      return;
    }

    if (s.activeTab === 'tabMeas') {
      const measSeries = (dimMeas, dimPred) => {
        const series = [];
        for (let sid = 0; sid < 12; sid++) {
          const buf = app.measHist.perStation[sid];
          const c = app.stationColors[sid];
          series.push(makeGappedLineSeries(buf, dimPred, c.replace('0.95', '0.35'), `pred ${sid + 1}`));
          series.push(makeDotsSeries(buf, dimMeas, c, `meas ${sid + 1}`));
        }
        return series;
      };

      app.plots.pltRho.draw(measSeries(0, 1));
      app.plots.pltRhoDot.draw(measSeries(2, 3));
      app.plots.pltPhi.draw(measSeries(4, 5));
      return;
    }

    if (s.activeTab === 'tabNees') {
      app.plots.pltNEES.draw([
        makeLineSeries(nn, 0, 'rgba(255,122,24,0.95)', 'NEES'),
        makeLineSeries(nn, 1, 'rgba(255,255,255,0.35)', 'lo'),
        makeLineSeries(nn, 2, 'rgba(255,255,255,0.35)', 'hi'),
      ]);
      app.plots.pltNIS.draw([
        makeLineSeries(nn, 3, 'rgba(16,185,129,0.95)', 'NIS'),
        makeLineSeries(nn, 4, 'rgba(255,255,255,0.35)', 'lo'),
        makeLineSeries(nn, 5, 'rgba(255,255,255,0.35)', 'hi'),
      ]);
      return;
    }

    // tabCtrl
    app.plots.pltUrt.draw([
      makeLineSeries(ctrl, 0, 'rgba(124,58,237,0.90)', 'u_r'),
      makeLineSeries(ctrl, 1, 'rgba(255,122,24,0.85)', 'u_t'),
    ]);

    app.plots.pltUm.draw([
      makeLineSeries(ctrl, 2, 'rgba(255,255,255,0.88)', '‖u‖'),
      makeHLine(() => app.state.umax, 'rgba(220,38,38,0.55)', 'u_max'),
    ]);

    // ctrl state plot: δr and δθ from estimate (computed on the fly)
    const tmpBuf = app._ctrlStateBuf;
    tmpBuf.len = 0;
    tmpBuf.idx = 0;

    est.forEachChronological((tt, data, off) => {
      const x = new Float64Array([data[off + 0], data[off + 1], data[off + 2], data[off + 3]]);
      const pol = eciToPolarDeviations(x, tt, app.state.ref);
      tmpBuf.push(tt, new Float64Array([pol.dr, pol.dtheta]));
    });

    app.plots.pltCtrlErr.draw([
      makeLineSeries(tmpBuf, 0, 'rgba(255,255,255,0.85)', 'δr'),
      makeLineSeries(tmpBuf, 1, 'rgba(255,122,24,0.85)', 'δθ'),
    ]);
  }

  function updateCtrlNotice() {
    const s = app.state;
    if (!app.ui.ctrlNotice) return;

    const ctrlEnabled = (s.ctrlMode === 'lqr' && controllerAllowed());
    const hasDisturb = s.disturb;
    const recentlyPerturbed = (s.t - s.lastPerturbT) < Math.max(30, 3 * s.dt);

    // show notice if controller off/disabled OR there is no disturbance/perturbation to react to
    const show = (!ctrlEnabled) || (!hasDisturb && !recentlyPerturbed);
    app.ui.ctrlNotice.style.display = show ? 'block' : 'none';

    if (show) {
      if (!controllerAllowed()) {
        app.ui.ctrlNoticeBody.textContent =
          'The controller is disabled because eccentricity is not zero. Set eccentricity to 0 to enable LQR, then use a disturbance or Perturb Orbit to see recovery.';
      } else if (!ctrlEnabled) {
        app.ui.ctrlNoticeBody.textContent =
          'The controller is currently OFF. Switch Controller type to LQR (eccentricity must be 0), then apply a disturbance or use Perturb Orbit to see correction.';
      } else if (!hasDisturb && !recentlyPerturbed) {
        app.ui.ctrlNoticeBody.textContent =
          'LQR is enabled, but there is no disturbance acting right now. Use Perturb Orbit to apply a small one-time velocity kick and watch the controller recover.';
      } else {
        app.ui.ctrlNoticeBody.textContent =
          'Turn on LQR (requires eccentricity = 0) and/or enable a disturbance. Or apply a one-time orbit perturbation to watch recovery.';
      }
    }
  }

  // ---------- UI wiring ----------
  function bindUI() {
    const ui = app.ui;

    ui.btnRunPause = document.getElementById('btnRunPause');
    ui.btnStep = document.getElementById('btnStep');
    ui.btnReset = document.getElementById('btnReset');
    ui.btnResetView = document.getElementById('btnResetView');
    ui.btnZoomCraft = document.getElementById('btnZoomCraft');

    ui.inpPerigeeAlt = document.getElementById('inpPerigeeAlt');
    ui.inpEcc = document.getElementById('inpEcc');
    ui.inpDt = document.getElementById('inpDt');

    ui.selMeasNoise = document.getElementById('selMeasNoise');
    ui.inpSigRho = document.getElementById('inpSigRho');
    ui.inpSigRhoDot = document.getElementById('inpSigRhoDot');
    ui.inpSigPhi = document.getElementById('inpSigPhi');

    ui.selDisturb = document.getElementById('selDisturb');
    ui.inpSigA = document.getElementById('inpSigA');

    ui.btnPerturbOrbit = document.getElementById('btnPerturbOrbit');
    ui.btnPerturbOrbit2 = document.getElementById('btnPerturbOrbit2');

    ui.selCtrl = document.getElementById('selCtrl');
    ui.inpUmaxG = document.getElementById('inpUmaxG');
    ui.ctrlEccNote = document.getElementById('ctrlEccNote');

    ui.selPlotWindow = document.getElementById('selPlotWindow');

    ui.orbitCanvas = document.getElementById('orbitCanvas');

    // relative error + covariance ellipse panel (optional)
    ui.ellipseCanvas = document.getElementById('ellipseCanvas');
    ui.ellipseHelp = document.getElementById('ellipseHelp');

    // HUD
    ui.hudTime = document.getElementById('hudTime');
    ui.hudStep = document.getElementById('hudStep');
    ui.hudDt = document.getElementById('hudDt');
    ui.hudVisible = document.getElementById('hudVisible');
    ui.hudMode = document.getElementById('hudMode');
    ui.hudSpeed = document.getElementById('hudSpeed');
    ui.hudNees = document.getElementById('hudNees');
    ui.hudNis = document.getElementById('hudNis');

    // pills
    ui.pillDisturb = document.getElementById('pillDisturb');
    ui.pillSat = document.getElementById('pillSat');
    ui.pillImpact = document.getElementById('pillImpact');

    // plots
    ui.pltX = document.getElementById('pltX');
    ui.pltY = document.getElementById('pltY');
    ui.pltXd = document.getElementById('pltXd');
    ui.pltYd = document.getElementById('pltYd');

    ui.pltErrX = document.getElementById('pltErrX');
    ui.pltErrY = document.getElementById('pltErrY');
    ui.pltErrXd = document.getElementById('pltErrXd');
    ui.pltErrYd = document.getElementById('pltErrYd');

    ui.pltRho = document.getElementById('pltRho');
    ui.pltRhoDot = document.getElementById('pltRhoDot');
    ui.pltPhi = document.getElementById('pltPhi');

    ui.pltNEES = document.getElementById('pltNEES');
    ui.pltNIS = document.getElementById('pltNIS');

    ui.pltUrt = document.getElementById('pltUrt');
    ui.pltUm = document.getElementById('pltUm');
    ui.pltCtrlErr = document.getElementById('pltCtrlErr');

    ui.ctrlSatBox = document.getElementById('ctrlSatBox');
    ui.ctrlSatText = document.getElementById('ctrlSatText');
    ui.ctrlSatFrac = document.getElementById('ctrlSatFrac');

    ui.stationBoard = document.getElementById('stationBoard');

    ui.ctrlNotice = document.getElementById('ctrlNotice');
    ui.ctrlNoticeBody = document.getElementById('ctrlNoticeBody');

    ui.speedButtons = Array.from(document.querySelectorAll('.speedbtn'));

    // tabs (supports both the "working" markup and the newer alternate markup)
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const panes = Array.from(document.querySelectorAll('.pane'));

    const activateTab = (targetId) => {
      if (!targetId) return;
      app.state.activeTab = targetId;

      // If panes exist, toggle via pane ids
      if (panes.length) {
        panes.forEach(p => p.classList.toggle('active', p.id === targetId));
      } else {
        // fallback: try explicit ids
        const ids = ['tabStates', 'tabErrors', 'tabMeas', 'tabNees', 'tabCtrl'];
        ids.forEach((id) => {
          const el = document.getElementById(id);
          if (el) el.classList.toggle('active', id === targetId);
        });
      }

      drawPlots();
    };

    if (tabs.length && tabs.some(b => b.dataset && b.dataset.tab)) {
      tabs.forEach((btn) => {
        btn.addEventListener('click', () => {
          tabs.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          activateTab(btn.dataset.tab);
        });
      });
    } else {
      // Alternate markup fallback: buttons with ids like "tabStatesBtn" or "tabStates"
      const alt = [
        { btn: document.getElementById('tabStatesBtn') || document.getElementById('tabStates'), tab: 'tabStates' },
        { btn: document.getElementById('tabErrorsBtn') || document.getElementById('tabErrors'), tab: 'tabErrors' },
        { btn: document.getElementById('tabMeasBtn')   || document.getElementById('tabMeas'),   tab: 'tabMeas'   },
        { btn: document.getElementById('tabNeesBtn')   || document.getElementById('tabNees'),   tab: 'tabNees'   },
        { btn: document.getElementById('tabCtrlBtn')   || document.getElementById('tabCtrl'),   tab: 'tabCtrl'   },
      ];
      alt.forEach(({ btn, tab }) => {
        if (!btn) return;
        btn.addEventListener('click', () => {
          alt.forEach(({ btn: b }) => b && b.classList.remove('active'));
          btn.classList.add('active');
          activateTab(tab);
        });
      });
    }

    ui.speedButtons.forEach((b) => {
      b.addEventListener('click', () => setSpeed(b.dataset.speed));
    });

    ui.btnRunPause.addEventListener('click', () => {
      app.state.running = !app.state.running;
      ui.btnRunPause.textContent = app.state.running ? 'Pause' : 'Run';
      app.clock.lastTs = 0;
      app.clock.accum = 0;
      updateHUD();
    });

    ui.btnStep.addEventListener('click', () => {
      if (app.state.running) return;
      stepOnce();
      app.orbitView.draw();
      drawPlots();
      updateHUD();
    });

    ui.btnReset.addEventListener('click', () => {
      app.state.running = false;
      ui.btnRunPause.textContent = 'Run';
      syncUiToState();
      resetSim();
      app.orbitView.draw();
      drawPlots();
    });

    ui.btnResetView.addEventListener('click', () => {
      app.orbitView.resetView();
      app.orbitView.draw();
    });

    ui.btnZoomCraft.addEventListener('click', () => {
      if (!app.orbitView.followCraft) app.orbitView.zoomToCraft();
      else {
        app.orbitView.followCraft = false;
        ui.btnZoomCraft.textContent = 'Zoom to Craft';
      }
      app.orbitView.draw();
    });

    ui.btnPerturbOrbit.addEventListener('click', () => { perturbOrbit(); updateCtrlNotice(); });
    ui.btnPerturbOrbit2.addEventListener('click', () => { perturbOrbit(); updateCtrlNotice(); });

    // live changes
    ui.inpEcc.addEventListener('input', () => { syncUiToState(); enforceControllerAvailability(); updateCtrlNotice(); });
    ui.selCtrl.addEventListener('change', () => { syncUiToState(); updateCtrlNotice(); });
    ui.selDisturb.addEventListener('change', () => { syncUiToState(); updateCtrlNotice(); });
    ui.selPlotWindow.addEventListener('change', () => { syncUiToState(); drawPlots(); });

    window.addEventListener('keydown', (e) => {
      if (e.key === ' ') {
        e.preventDefault();
        ui.btnRunPause.click();
      } else if (e.key.toLowerCase() === 'r') {
        ui.btnReset.click();
      } else if (e.key.toLowerCase() === 's') {
        ui.btnStep.click();
      }
    });
  }

  // ---------- Main loop (real-time scaled) ----------
  function tick(ts) {
    app.animFrame++;
    const s = app.state;

    if (s.running) {
      if (!app.clock.lastTs) app.clock.lastTs = ts;
      const dtReal = (ts - app.clock.lastTs) / 1000;
      app.clock.lastTs = ts;

      // accumulate simulation time scaled by speed
      const add = clamp(dtReal, 0, 0.25) * s.speed;
      app.clock.accum += add;

      // avoid spiral-of-death
      const maxStepsThisFrame = 600;
      let steps = 0;
      while (app.clock.accum >= s.dt && steps < maxStepsThisFrame) {
        stepOnce();
        app.clock.accum -= s.dt;
        steps++;
      }
    }

    app.orbitView.draw();
    drawPlots();
    if (app.relPosPlot) app.relPosPlot.draw();
    updateHUD();
    requestAnimationFrame(tick);
  }

  // ---------- Init ----------
  function init() {
    initColors();
    app.rng = XorShift32(1234567);

    bindUI();
    app.orbitView = new OrbitView(app.ui.orbitCanvas);

    // Optional: relative error + covariance ellipses panel
    if (app.ui.ellipseCanvas) app.relPosPlot = new RelPosCovPlot(app.ui.ellipseCanvas);

    // small helper buffer used for ctrl-state plotting
    app._ctrlStateBuf = new RingBuffer(45000, 2);

    createStationBoard();
    setupPlots();

    // start speed default to 10x button state
    setSpeed(10);

    // set default window to 3h already in HTML; sync it
    syncUiToState();

    resetSim();
    app.orbitView.draw();
    drawPlots();
    updateHUD();

    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
