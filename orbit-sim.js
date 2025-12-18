/* orbit-sim.js
   2D Orbit Determination + EKF Demo
   Self-contained: physics, estimator, rendering, UI.
   Author: Carissa Mayo (Portfolio Demo)
*/

/* ============================================================
   Minimal numeric helpers (NO external libs)
   ============================================================ */
const numeric = {
  identity(n) {
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );
  },
  transpose(A) {
    return A[0].map((_, j) => A.map(r => r[j]));
  },
  add(A, B) {
    if (Array.isArray(A[0])) return A.map((r, i) => r.map((v, j) => v + B[i][j]));
    return A.map((v, i) => v + B[i]);
  },
  sub(A, B) {
    if (Array.isArray(A[0])) return A.map((r, i) => r.map((v, j) => v - B[i][j]));
    return A.map((v, i) => v - B[i]);
  },
  mul(A, s) {
    if (Array.isArray(A[0])) return A.map(r => r.map(v => v * s));
    return A.map(v => v * s);
  },
  dotMMsmall(A, B) {
    const n = A.length, m = B[0].length, p = B.length;
    const C = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < p; k++) {
        const aik = A[i][k];
        for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j];
      }
    }
    return C;
  },
  dotMV(A, v) {
    return A.map(r => r.reduce((s, aij, j) => s + aij * v[j], 0));
  },
  // Gauss–Jordan inverse (robust enough for small/medium matrices)
  inv(M) {
    const n = M.length;
    const A = M.map(r => r.slice());
    const I = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );

    for (let col = 0; col < n; col++) {
      // pivot
      let pivRow = col;
      let pivVal = Math.abs(A[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(A[r][col]);
        if (v > pivVal) { pivVal = v; pivRow = r; }
      }
      if (!isFinite(pivVal) || pivVal === 0) throw new Error("numeric.inv: singular");

      if (pivRow !== col) {
        [A[col], A[pivRow]] = [A[pivRow], A[col]];
        [I[col], I[pivRow]] = [I[pivRow], I[col]];
      }

      const piv = A[col][col];
      for (let j = 0; j < n; j++) { A[col][j] /= piv; I[col][j] /= piv; }

      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = A[r][col];
        if (f === 0) continue;
        for (let j = 0; j < n; j++) {
          A[r][j] -= f * A[col][j];
          I[r][j] -= f * I[col][j];
        }
      }
    }
    return I;
  },
  diag(arr) {
    const n = arr.length;
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? arr[i] : 0))
    );
  }
};

/* ============================================================
   Global constants and helpers
   ============================================================ */
const MU = 398600;              // km^3/s^2
const R_E = 6378;               // km
const OMEGA_E = 7.2722052e-5;   // rad/s
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }
function dot2(a, b) { return a[0] * b[0] + a[1] * b[1]; }
function norm2(v) { return Math.hypot(v[0], v[1]); }
function add2(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
function sub2(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
function scale2(v, s) { return [v[0] * s, v[1] * s]; }
function add4(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]; }
function scale4(v, s) { return [v[0] * s, v[1] * s, v[2] * s, v[3] * s]; }

function randn() {
  // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/* ============================================================
   Simulation state
   ============================================================ */
const app = {
  // sim
  running: false,
  t: 0,
  dt: 1,
  stepCount: 0,
  speed: 1,
  maxSteps: 1000000,

  // truth accel control (km/s^2)
  u: [0, 0],

  // disturbances + measurement noise
  procNoiseOn: false,
  measNoiseOn: true,
  procModel: "gm",     // "gm" or "white"
  procSigma: 5e-8,     // km/s^2
  procTau: 300,        // s
  procBias: [0, 0],    // accel bias (GM)

  // filter tuning
  alpha: 0.05,
  epsFD: 1e-6,
  gateMode: "none",
  procModelEKF: "2body",
  crashMode: "reset",

  // states
  xTrue: [7000, 0, 0, 7.5],
  xEst: [6999, 0, 0, 7.5],
  P: null,

  // stations
  stations: [],
  visibleStations: [],

  // history
  history: null,

  // canvases
  orbit: { canvas: null, ctx: null },
  plots: {},

  // orbit view camera
  view: {
    pxPerKm: 0.03,
    centerKm: [0, 0],
    dragging: false,
    last: [0, 0]
  },

  // station UI handles
  stationUI: [],

  // dom cache
  el: {}
};

function ensureHistory() {
  if (app.history) return;
  app.history = {
    t: [],
    xT: [],  // [X,Xd,Y,Yd]
    xE: [],
    nees: [],
    nis: [],
    nisDof: [],
    meas: [] // {t, stationIdx, y:[rho,rhod,phi], yhat:[...], innov:[...]}
  };
}

/* ============================================================
   Dynamics + RK4 truth propagation
   ============================================================ */
function derivTwoBody(x, u = [0, 0]) {
  const X = x[0], Xd = x[1], Y = x[2], Yd = x[3];
  const r = Math.hypot(X, Y);
  const acc = -MU / (r * r * r);
  return [Xd, acc * X + u[0], Yd, acc * Y + u[1]];
}

function rk4Step(x, u, dt) {
  const k1 = derivTwoBody(x, u);
  const x2 = add4(x, scale4(k1, dt / 2));
  const k2 = derivTwoBody(x2, u);
  const x3 = add4(x, scale4(k2, dt / 2));
  const k3 = derivTwoBody(x3, u);
  const x4 = add4(x, scale4(k3, dt));
  const k4 = derivTwoBody(x4, u);

  const out = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    out[i] = x[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return out;
}

/* ============================================================
   Disturbance models
   ============================================================ */
function updateProcessNoise(dt) {
  if (!app.procNoiseOn) {
    app.procBias[0] = 0; app.procBias[1] = 0;
    return [0, 0];
  }

  if (app.procModel === "white") {
    return [randn() * app.procSigma, randn() * app.procSigma];
  }

  // Gauss–Markov (first-order) accel bias
  const tau = Math.max(1e-6, app.procTau);
  const a = Math.exp(-dt / tau);
  const q = Math.sqrt(1 - a * a) * app.procSigma;
  app.procBias[0] = a * app.procBias[0] + q * randn();
  app.procBias[1] = a * app.procBias[1] + q * randn();
  return [app.procBias[0], app.procBias[1]];
}

/* ============================================================
   Stations + measurement model
   ============================================================ */
function initStations() {
  // Distinct hues; keep count manageable for readability
  const N = 6;
  const stas = [];
  for (let i = 0; i < N; i++) {
    const lon0 = (i / N) * 2 * Math.PI;
    const hue = (i * 360) / N;
    stas.push({
      lon0,
      name: `GS-${i + 1}`,
      color: `hsl(${hue}, 80%, 55%)`
    });
  }
  app.stations = stas;
}

function stationPositions(t) {
  return app.stations.map(st => {
    const lon = st.lon0 + OMEGA_E * t;
    return [R_E * Math.cos(lon), R_E * Math.sin(lon)];
  });
}

// Visibility: above horizon if dot(r_s, r - r_s) > 0
function visibleStations(x) {
  const pos = [x[0], x[2]];
  const stations = stationPositions(app.t);
  const out = [];
  for (let i = 0; i < stations.length; i++) {
    const r_s = stations[i];
    const rho = sub2(pos, r_s);
    const vis = dot2(r_s, rho) > 0;
    if (vis) out.push({ idx: i, r_s });
  }
  return out;
}

// y = [rho, rhodot, phi]
function measureStation(x, r_s) {
  const pos = [x[0], x[2]];
  const vel = [x[1], x[3]];
  const rhoVec = sub2(pos, r_s);
  const rho = norm2(rhoVec);
  const rhod = dot2(rhoVec, vel) / Math.max(1e-12, rho);
  const phi = Math.atan2(pos[1] - r_s[1], pos[0] - r_s[0]);
  return [rho, rhod, phi];
}

/* ============================================================
   EKF: discrete-time via RK4 + numerical linearization
   ============================================================ */
function dynStep(x, u, dt) {
  return rk4Step(x, u, dt);
}

function dynJacobianFD(x, u, dt, eps) {
  // F = d f_discrete / d x  (4x4)
  const F = Array.from({ length: 4 }, () => Array(4).fill(0));
  const f0 = dynStep(x, u, dt);

  for (let j = 0; j < 4; j++) {
    const xp = x.slice();
    xp[j] += eps;
    const fp = dynStep(xp, u, dt);
    for (let i = 0; i < 4; i++) F[i][j] = (fp[i] - f0[i]) / eps;
  }
  return F;
}

function ekfPredict(x, P, u, Q, dt) {
  const xp = dynStep(x, u, dt);
  const F = dynJacobianFD(x, u, dt, app.epsFD);
  const Ft = numeric.transpose(F);
  const Pp = numeric.add(numeric.dotMMsmall(numeric.dotMMsmall(F, P), Ft), Q);
  return { x: xp, P: Pp, F };
}

// Measurement Jacobian H: (3*Ns) x 4
function measJacobian(x, stations) {
  const eps = app.epsFD;
  const m = 3 * stations.length;
  const H = Array.from({ length: m }, () => Array(4).fill(0));

  const h0 = [];
  for (const st of stations) h0.push(...measureStation(x, st.r_s));

  for (let j = 0; j < 4; j++) {
    const xp = x.slice();
    xp[j] += eps;

    const hp = [];
    for (const st of stations) hp.push(...measureStation(xp, st.r_s));

    for (let i = 0; i < m; i++) H[i][j] = (hp[i] - h0[i]) / eps;
  }
  return H;
}

function ekfUpdate(x, P, yObs, stations, R) {
  if (!stations.length || !yObs.length) return { x, P, yhat: [], innov: [], S: null };

  const H = measJacobian(x, stations);

  // predicted measurement yhat
  const yhat = [];
  for (const st of stations) yhat.push(...measureStation(x, st.r_s));

  // innovation
  const innov = numeric.sub(yObs, yhat);

  // normalize angle innovations to [-pi, pi] for each station's phi (3rd component)
  for (let k = 0; k < stations.length; k++) {
    const idxPhi = 3 * k + 2;
    let a = innov[idxPhi];
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    innov[idxPhi] = a;
  }

  const Ht = numeric.transpose(H);
  const S = numeric.add(numeric.dotMMsmall(numeric.dotMMsmall(H, P), Ht), R);

  // Optional NIS gating (very conservative): reject full update if NIS too large
  let nis = NaN;
  try {
    const Sinv = numeric.inv(S);
    const tmp = numeric.dotMV(Sinv, innov);
    nis = innov.reduce((s, v, i) => s + v * tmp[i], 0);

    if (app.gateMode === "nis") {
      const dof = innov.length;
      const hi = chi2_ppf(1 - app.alpha / 2, dof);
      if (nis > hi) {
        return { x, P, yhat, innov, S, gated: true, nis };
      }
    }
  } catch { /* leave */ }

  const K = numeric.dotMMsmall(numeric.dotMMsmall(P, Ht), numeric.inv(S));
  const xNew = numeric.add(x, numeric.dotMV(K, innov));

  const I = numeric.identity(4);
  const KH = numeric.dotMMsmall(K, H);
  const PNew = numeric.dotMMsmall(numeric.sub(I, KH), P);

  return { x: xNew, P: PNew, yhat, innov, S, gated: false, nis };
}

/* ============================================================
   Consistency bounds (chi-square ppf) via Wilson–Hilferty
   ============================================================ */
function normal_ppf(p) {
  // Acklam approximation
  const a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00];
  const b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00];
  const d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
              3.754408661907416e+00];

  const plow = 0.02425;
  const phigh = 1 - plow;

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
  r = q * q;
  return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
         (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
}

function chi2_ppf(p, k) {
  // Wilson–Hilferty
  const z = normal_ppf(p);
  const a = 2 / (9 * k);
  const x = k * Math.pow(1 - a + z * Math.sqrt(a), 3);
  return Math.max(0, x);
}

/* ============================================================
   Crash guard
   ============================================================ */
function checkCrash() {
  const rT = Math.hypot(app.xTrue[0], app.xTrue[2]);
  const rE = Math.hypot(app.xEst[0], app.xEst[2]);

  if (rT < R_E || rE < R_E) {
    if (app.crashMode === "reset") resetSim();
    else app.running = false;
  }
}

/* ============================================================
   UI + station boards
   ============================================================ */
function buildStationBoards() {
  const boardMain = app.el.stationBoard;
  const boardSmall = app.el.stationBoardSmall;

  boardMain.innerHTML = "";
  boardSmall.innerHTML = "";
  app.stationUI = [];

  for (let i = 0; i < app.stations.length; i++) {
    const st = app.stations[i];

    const makeTile = (compact) => {
      const tile = document.createElement("div");
      tile.className = "stationtile";
      tile.style.borderColor = st.color;

      tile.innerHTML = `
        <div class="stationtile__swatch" style="background:${st.color}"></div>
        <div class="stationtile__dish" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 1 10 10h-2a8 8 0 0 0-8-8V2z"/></svg>
        </div>
        <div class="stationtile__name">${st.name}</div>
        <div class="stationtile__meta"><span class="mono">${compact ? "" : "ID " + (i+1)}</span></div>
      `;

      // Compact tiles hide name/meta via CSS in strip, but keep DOM consistent.
      return tile;
    };

    const tileMain = makeTile(true);
    const tileSmall = makeTile(false);

    boardMain.appendChild(tileMain);
    boardSmall.appendChild(tileSmall);

    app.stationUI.push({
      tileMain,
      tileSmall,
      dishMain: tileMain.querySelector(".stationtile__dish"),
      dishSmall: tileSmall.querySelector(".stationtile__dish"),
      color: st.color
    });
  }
}

function updateStationBoardsAndDishes() {
  const visSet = new Set(app.visibleStations.map(v => v.idx));
  const stPos = stationPositions(app.t);
  const craft = [app.xTrue[0], app.xTrue[2]];

  for (let i = 0; i < app.stationUI.length; i++) {
    const ui = app.stationUI[i];
    const measuring = visSet.has(i);
    ui.tileMain.classList.toggle("is-measuring", measuring);
    ui.tileSmall.classList.toggle("is-measuring", measuring);

    // rotate dish icons toward craft (screen rotation; match orbit view)
    const s = stPos[i];
    const ang = Math.atan2(craft[1] - s[1], craft[0] - s[0]);
    ui.dishMain.style.transform = `rotate(${ang}rad)`;
    ui.dishSmall.style.transform = `rotate(${ang}rad)`;
  }
}

/* ============================================================
   Orbit canvas: pan/zoom + rendering
   ============================================================ */
function setupOrbitCanvas() {
  app.orbit.canvas = app.el.orbitCanvas;
  app.orbit.ctx = app.orbit.canvas.getContext("2d");

  const resize = () => {
    const c = app.orbit.canvas;
    c.width = c.clientWidth;
    c.height = c.clientHeight;

    // default scale fits Earth nicely
    const w = c.width, h = c.height;
    const target = 0.38 * Math.min(w, h);
    app.view.pxPerKm = target / R_E;
  };

  resize();
  window.addEventListener("resize", resize);

  // Pan/zoom interaction
  const view = app.view;
  app.orbit.canvas.addEventListener("mousedown", (e) => {
    view.dragging = true;
    view.last = [e.clientX, e.clientY];
  });
  window.addEventListener("mouseup", () => { view.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!view.dragging) return;
    const dx = e.clientX - view.last[0];
    const dy = e.clientY - view.last[1];
    view.last = [e.clientX, e.clientY];

    // screen dx -> km shift (note y flipped in world draw)
    const kmPerPx = 1 / view.pxPerKm;
    view.centerKm[0] -= dx * kmPerPx;
    view.centerKm[1] += dy * kmPerPx;
  }, { passive: true });

  app.orbit.canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.10 : 0.90;
    view.pxPerKm = clamp(view.pxPerKm * factor, 1e-4, 0.5);
  }, { passive: false });
}

function worldToScreen(xKm, yKm) {
  const c = app.orbit.canvas;
  const w = c.width, h = c.height;
  const px = app.view.pxPerKm;

  const cx = w / 2;
  const cy = h / 2;
  const dx = (xKm - app.view.centerKm[0]) * px;
  const dy = (yKm - app.view.centerKm[1]) * px;

  // screen y down, but world y up
  return [cx + dx, cy - dy];
}

function drawCovarianceEllipse(ctx) {
  // Use position covariance (X,Y) -> indices (0,2)
  const Pxx = app.P[0][0];
  const Pxy = app.P[0][2];
  const Pyy = app.P[2][2];

  // Eigen of 2x2
  const tr = Pxx + Pyy;
  const det = Pxx * Pyy - Pxy * Pxy;
  const disc = Math.max(0, tr * tr / 4 - det);
  const s = Math.sqrt(disc);

  const l1 = tr / 2 + s;
  const l2 = tr / 2 - s;

  if (!(l1 > 0) || !(l2 > 0)) return;

  // eigenvector angle
  const ang = 0.5 * Math.atan2(2 * Pxy, Pxx - Pyy);
  const sig = 2; // 2σ ellipse

  const a = sig * Math.sqrt(l1); // km
  const b = sig * Math.sqrt(l2); // km

  const center = worldToScreen(app.xEst[0], app.xEst[2]);
  const pxPerKm = app.view.pxPerKm;

  ctx.save();
  ctx.translate(center[0], center[1]);
  ctx.rotate(-ang); // because screen y is inverted vs world
  ctx.beginPath();
  ctx.ellipse(0, 0, a * pxPerKm, b * pxPerKm, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(37,99,235,0.65)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawSignalLink(ctx, s, craft, color) {
  const [sx, sy] = worldToScreen(s[0], s[1]);
  const [cx, cy] = worldToScreen(craft[0], craft[1]);

  // solid LOS line
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(cx, cy);
  ctx.stroke();

  // animated squiggle “data” link along the same segment
  const dx = cx - sx;
  const dy = cy - sy;
  const L = Math.hypot(dx, dy);
  if (L > 1) {
    const ux = dx / L;
    const uy = dy / L;
    const nx = -uy;
    const ny = ux;

    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    const phase = app.t * 2.5;
    const amp = 6;
    const step = 16;
    for (let d = 0; d <= L; d += step) {
      const w = amp * Math.sin(0.06 * d + phase);
      const px = sx + ux * d + nx * w;
      const py = sy + uy * d + ny * w;
      if (d === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawOrbitView() {
  const ctx = app.orbit.ctx;
  const c = app.orbit.canvas;
  const w = c.width, h = c.height;

  ctx.clearRect(0, 0, w, h);

  // Earth disk (space background handled by CSS)
  const [ex, ey] = worldToScreen(0, 0);
  const rPx = R_E * app.view.pxPerKm;

  // Earth
  ctx.save();
  ctx.beginPath();
  ctx.arc(ex, ey, rPx, 0, Math.PI * 2);
  const grad = ctx.createRadialGradient(ex - 0.2 * rPx, ey + 0.2 * rPx, 0.2 * rPx, ex, ey, rPx);
  grad.addColorStop(0, "rgba(96,165,250,0.95)");
  grad.addColorStop(1, "rgba(6,11,22,0.95)");
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.restore();

  // Stations
  const stPos = stationPositions(app.t);
  for (let i = 0; i < stPos.length; i++) {
    const st = app.stations[i];
    const s = stPos[i];
    const [sx, sy] = worldToScreen(s[0], s[1]);

    ctx.save();
    ctx.fillStyle = st.color;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Visible station links (signal)
  const craft = [app.xTrue[0], app.xTrue[2]];
  for (const v of app.visibleStations) {
    drawSignalLink(ctx, stPos[v.idx], craft, app.stations[v.idx].color);
  }

  // Truth marker
  {
    const [tx, ty] = worldToScreen(app.xTrue[0], app.xTrue[2]);
    ctx.save();
    ctx.fillStyle = "rgba(249,115,22,0.95)";
    ctx.beginPath();
    ctx.arc(tx, ty, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Estimate marker
  {
    const [hx, hy] = worldToScreen(app.xEst[0], app.xEst[2]);
    ctx.save();
    ctx.strokeStyle = "rgba(37,99,235,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(hx, hy, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Covariance ellipse
  if (app.P) drawCovarianceEllipse(ctx);

  // Update station boards / dish rotation
  updateStationBoardsAndDishes();
}

/* ============================================================
   Plot interactions (paused-only pan/zoom)
   ============================================================ */
function enablePlotInteractions() {
  const ids = ["plotStates", "plotErrors", "plotMeas", "plotNEES"];
  for (const id of ids) {
    const canvas = app.el[id];
    app.plots[id] = {
      el: canvas,
      ctx: canvas.getContext("2d"),
      scale: 1,
      offsetX: 0,
      offsetY: 0,
      dragging: false,
      last: [0, 0]
    };

    // size to CSS
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    setupPlotPanZoom(app.plots[id]);
  }

  window.addEventListener("resize", () => {
    for (const id of ids) {
      const c = app.plots[id].el;
      c.width = c.clientWidth;
      c.height = c.clientHeight;
    }
  });
}

function setupPlotPanZoom(plot) {
  plot.el.addEventListener("mousedown", (e) => {
    if (app.running) return;
    plot.dragging = true;
    plot.last = [e.offsetX, e.offsetY];
  });
  window.addEventListener("mouseup", () => { plot.dragging = false; });
  plot.el.addEventListener("mousemove", (e) => {
    if (!plot.dragging || app.running) return;
    const dx = e.offsetX - plot.last[0];
    const dy = e.offsetY - plot.last[1];
    plot.last = [e.offsetX, e.offsetY];
    plot.offsetX += dx;
    plot.offsetY += dy;
  });
  plot.el.addEventListener("wheel", (e) => {
    if (app.running) return;
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    plot.scale = clamp(plot.scale * factor, 0.4, 6.0);
  }, { passive: false });
}

/* ============================================================
   Plot drawing
   - Running: auto-window last 300s, reset pan/zoom
   - Paused: allow pan/zoom
   ============================================================ */
function drawPlots() {
  ensureHistory();
  const H = app.history;
  if (!H.t.length) return;

  const tMax = H.t[H.t.length - 1];
  const tMin = app.running ? Math.max(0, tMax - 300) : H.t[0];

  // reset plot transforms when running (locks to autoscale)
  if (app.running) {
    for (const k in app.plots) {
      app.plots[k].scale = 1;
      app.plots[k].offsetX = 0;
      app.plots[k].offsetY = 0;
    }
  }

  // index range for time window
  let i0 = 0;
  while (i0 < H.t.length && H.t[i0] < tMin) i0++;
  const i1 = H.t.length - 1;

  const drawFrame = (plot, title) => {
    const ctx = plot.ctx;
    const w = ctx.canvas.width, h = ctx.canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    // title
    ctx.fillStyle = "#64748b";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    ctx.fillText(title, 10, 16);

    // border
    ctx.strokeStyle = "rgba(226,232,240,0.95)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // plot rect
    return { x: 46, y: 28, w: w - 58, h: h - 44 };
  };

  const applyPanZoom = (plot, px, py, w, h) => {
    if (app.running) return [px, py];
    const cx = w * 0.5;
    const cy = h * 0.5;
    const sx = (px - cx) * plot.scale + cx + plot.offsetX;
    const sy = (py - cy) * plot.scale + cy + plot.offsetY;
    return [sx, sy];
  };

  const xToPx = (t, rect) => rect.x + ((t - tMin) / Math.max(1e-9, (tMax - tMin))) * rect.w;

  const yToPx = (y, rect, yMin, yMax) =>
    rect.y + rect.h - ((y - yMin) / Math.max(1e-12, (yMax - yMin))) * rect.h;

  const minmax = (getY) => {
    let mn = Infinity, mx = -Infinity;
    for (let i = i0; i <= i1; i++) {
      const v = getY(i);
      if (!isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!isFinite(mn) || !isFinite(mx)) { mn = -1; mx = 1; }
    if (mn === mx) { mn -= 1; mx += 1; }
    const pad = 0.08 * (mx - mn);
    return { mn: mn - pad, mx: mx + pad };
  };

  const drawGrid = (ctx, rect) => {
    ctx.save();
    ctx.strokeStyle = "rgba(148,163,184,0.18)";
    ctx.lineWidth = 1;

    const nx = 6, ny = 5;
    for (let i = 0; i <= nx; i++) {
      const x = rect.x + (i / nx) * rect.w;
      ctx.beginPath();
      ctx.moveTo(x, rect.y);
      ctx.lineTo(x, rect.y + rect.h);
      ctx.stroke();
    }
    for (let j = 0; j <= ny; j++) {
      const y = rect.y + (j / ny) * rect.h;
      ctx.beginPath();
      ctx.moveTo(rect.x, y);
      ctx.lineTo(rect.x + rect.w, y);
      ctx.stroke();
    }
    ctx.restore();
  };

  const clipToRect = (ctx, rect) => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();
  };

  const unclip = (ctx) => ctx.restore();

  const plotLine = (plot, rect, yMin, yMax, getY, stroke, dash = null) => {
    const ctx = plot.ctx;
    const w = ctx.canvas.width, h = ctx.canvas.height;

    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.8;
    ctx.beginPath();

    let started = false;
    for (let i = i0; i <= i1; i++) {
      const t = H.t[i];
      const y = getY(i);
      if (!isFinite(y)) continue;

      let px = xToPx(t, rect);
      let py = yToPx(y, rect, yMin, yMax);
      [px, py] = applyPanZoom(plot, px, py, w, h);

      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  };

  const drawLegend = (ctx, items) => {
    ctx.save();
    ctx.font = "12px var(--sans)";
    let x = 10, y = ctx.canvas.height - 10;
    for (const it of items) {
      ctx.fillStyle = it.color;
      ctx.fillRect(x, y - 10, 10, 10);
      ctx.fillStyle = "#334155";
      ctx.fillText(it.label, x + 14, y - 1);
      x += 14 + ctx.measureText(it.label).width + 16;
    }
    ctx.restore();
  };

  // ---------------- States plot ----------------
  {
    const plot = app.plots.plotStates;
    const ctx = plot.ctx;
    const rect = drawFrame(plot, "States (X,Y) — truth vs estimate");
    drawGrid(ctx, rect);
    clipToRect(ctx, rect);

    const mmX = minmax(i => H.xE[i][0]);
    const mmY = minmax(i => H.xE[i][2]);

    plotLine(plot, rect, mmX.mn, mmX.mx, i => H.xT[i][0], "rgba(249,115,22,0.65)");
    plotLine(plot, rect, mmX.mn, mmX.mx, i => H.xE[i][0], "rgba(37,99,235,0.95)");
    plotLine(plot, rect, mmY.mn, mmY.mx, i => H.xT[i][2], "rgba(249,115,22,0.35)", [6, 4]);
    plotLine(plot, rect, mmY.mn, mmY.mx, i => H.xE[i][2], "rgba(37,99,235,0.60)", [6, 4]);

    unclip(ctx);
    drawLegend(ctx, [
      { label: "Est X", color: "rgba(37,99,235,0.95)" },
      { label: "True X", color: "rgba(249,115,22,0.65)" },
      { label: "Est Y (dash)", color: "rgba(37,99,235,0.60)" },
      { label: "True Y (dash)", color: "rgba(249,115,22,0.35)" }
    ]);
  }

  // ---------------- Errors plot ----------------
  {
    const plot = app.plots.plotErrors;
    const ctx = plot.ctx;
    const rect = drawFrame(plot, "Errors (X,Y) — estimate minus truth");
    drawGrid(ctx, rect);
    clipToRect(ctx, rect);

    const mm = minmax(i => H.xE[i][0] - H.xT[i][0]);

    plotLine(plot, rect, mm.mn, mm.mx, i => (H.xE[i][0] - H.xT[i][0]), "rgba(37,99,235,0.95)");
    plotLine(plot, rect, mm.mn, mm.mx, i => (H.xE[i][2] - H.xT[i][2]), "rgba(249,115,22,0.90)", [6, 4]);

    unclip(ctx);
    drawLegend(ctx, [
      { label: "X err", color: "rgba(37,99,235,0.95)" },
      { label: "Y err (dash)", color: "rgba(249,115,22,0.90)" }
    ]);
  }

  // ---------------- Measurements plot ----------------
  // 3 stacked panels: rho, rhodot, phi; points colored by station
  {
    const plot = app.plots.plotMeas;
    const ctx = plot.ctx;
    const rectFull = drawFrame(plot, "Measurements — points by station (measured vs predicted)");
    drawGrid(ctx, rectFull);

    const subH = (rectFull.h - 20) / 3;
    const panels = [
      { name: "rho (km)",    idx: 0, rect: { x: rectFull.x, y: rectFull.y,              w: rectFull.w, h: subH } },
      { name: "rhod (km/s)", idx: 1, rect: { x: rectFull.x, y: rectFull.y + subH + 10, w: rectFull.w, h: subH } },
      { name: "phi (rad)",   idx: 2, rect: { x: rectFull.x, y: rectFull.y + 2*(subH+10), w: rectFull.w, h: subH } }
    ];

    const meas = H.meas;
    if (meas.length) {
      // Build y extents from the windowed measurement records
      const inWindow = [];
      for (let k = meas.length - 1; k >= 0; k--) {
        if (meas[k].t < tMin) break;
        inWindow.push(meas[k]);
      }

      for (const p of panels) {
        const rect = p.rect;

        // local y-limits
        let mn = Infinity, mx = -Infinity;
        for (const m of inWindow) {
          const v = m.y[p.idx];
          const vh = m.yhat[p.idx];
          if (isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
          if (isFinite(vh)) { mn = Math.min(mn, vh); mx = Math.max(mx, vh); }
        }
        if (!isFinite(mn) || !isFinite(mx)) { mn = -1; mx = 1; }
        if (mn === mx) { mn -= 1; mx += 1; }
        const pad = 0.08 * (mx - mn);
        const yMin = mn - pad, yMax = mx + pad;

        // panel title
        ctx.save();
        ctx.fillStyle = "rgba(100,116,139,0.9)";
        ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
        ctx.fillText(p.name, rect.x + 4, rect.y + 12);
        ctx.restore();

        // clip and draw points
        clipToRect(ctx, rect);

        // predicted = small line segments per station (light)
        // measured = dots
        for (const m of inWindow) {
          const st = app.stations[m.stationIdx];
          const color = st ? st.color : "rgba(37,99,235,0.9)";

          const px0 = xToPx(m.t, rect);
          const pyMeas = yToPx(m.y[p.idx], rect, yMin, yMax);
          const pyHat = yToPx(m.yhat[p.idx], rect, yMin, yMax);

          const [px, pyM] = applyPanZoom(plot, px0, pyMeas, ctx.canvas.width, ctx.canvas.height);
          const [pxh, pyH] = applyPanZoom(plot, px0, pyHat, ctx.canvas.width, ctx.canvas.height);

          // predicted mark
          ctx.save();
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.35;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(pxh - 3, pyH);
          ctx.lineTo(pxh + 3, pyH);
          ctx.stroke();
          ctx.restore();

          // measured point
          ctx.save();
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(px, pyM, 2.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }

        unclip(ctx);

        // panel border
        ctx.save();
        ctx.strokeStyle = "rgba(226,232,240,0.9)";
        ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
        ctx.restore();
      }
    }
  }

  // ---------------- NEES / NIS plot ----------------
  {
    const plot = app.plots.plotNEES;
    const ctx = plot.ctx;
    const rect = drawFrame(plot, "Consistency — NEES and NIS with χ² bounds");
    drawGrid(ctx, rect);
    clipToRect(ctx, rect);

    const mm = minmax(i => H.nees[i]);

    // bounds
    const dofNees = 4;
    const loNEES = chi2_ppf(app.alpha / 2, dofNees);
    const hiNEES = chi2_ppf(1 - app.alpha / 2, dofNees);

    // plot bounds
    plotLine(plot, rect, mm.mn, mm.mx, _ => loNEES, "rgba(148,163,184,0.70)", [4, 6]);
    plotLine(plot, rect, mm.mn, mm.mx, _ => hiNEES, "rgba(148,163,184,0.70)", [4, 6]);

    plotLine(plot, rect, mm.mn, mm.mx, i => H.nees[i], "rgba(249,115,22,0.95)");

    // NIS plotted on same axis for simplicity (it’s still useful shape-wise)
    plotLine(plot, rect, mm.mn, mm.mx, i => H.nis[i], "rgba(37,99,235,0.90)", [6, 4]);

    unclip(ctx);
    drawLegend(ctx, [
      { label: "NEES", color: "rgba(249,115,22,0.95)" },
      { label: "NIS (dash)", color: "rgba(37,99,235,0.90)" },
      { label: "χ² bounds", color: "rgba(148,163,184,0.70)" }
    ]);
  }
}

/* ============================================================
   Measurements generation + EKF integration
   ============================================================ */
function makeMeasurements() {
  const vis = visibleStations(app.xTrue);
  app.visibleStations = vis;

  const y = [];
  const Rdiag = [];

  const sigRho = parseFloat(app.el.inpRtrueRho.value);
  const sigRhod = parseFloat(app.el.inpRtrueRhod.value);
  const sigPhi = parseFloat(app.el.inpRtruePhi.value);

  for (const { idx, r_s } of vis) {
    const y0 = measureStation(app.xTrue, r_s);
    const n = app.measNoiseOn ? [randn() * sigRho, randn() * sigRhod, randn() * sigPhi] : [0, 0, 0];

    const yy = [y0[0] + n[0], y0[1] + n[1], y0[2] + n[2]];
    y.push(...yy);

    Rdiag.push(sigRho * sigRho, sigRhod * sigRhod, sigPhi * sigPhi);
  }

  const R = numeric.diag(Rdiag.length ? Rdiag : [1, 1, 1]);
  return { y, R, vis };
}

function simStepFull() {
  ensureHistory();

  // dt from UI (guarded)
  const dt = clamp(parseFloat(app.el.inpDt.value) || 1, 0.01, 60);
  app.dt = dt;

  // max steps
  app.maxSteps = Math.max(1000, parseInt(app.el.inpSteps.value || "1000000", 10));

  // stop if max steps reached
  if (app.stepCount >= app.maxSteps) {
    app.running = false;
    app.el.statusRun.textContent = "PAUSED";
    app.el.btnToggleRun.textContent = "Run";
    return;
  }

  // truth propagation (with disturbance)
  const aDist = updateProcessNoise(dt);
  const uTot = add2(app.u, aDist);
  app.xTrue = rk4Step(app.xTrue, uTot, dt);

  // measurements
  const meas = makeMeasurements();

  // EKF prediction
  // Q: simple accel perturbation mapped into state with dt^2 / dt
  const qAcc = 1e-10; // conservative filter process noise (km/s^2)^2
  const G = [
    [0.5 * dt * dt, 0],
    [dt, 0],
    [0, 0.5 * dt * dt],
    [0, dt]
  ];
  const Qacc = [
    [qAcc, 0],
    [0, qAcc]
  ];
  // Q = G Qacc G'
  const GQ = numeric.dotMMsmall(G, Qacc);
  const Q = numeric.dotMMsmall(GQ, numeric.transpose(G));

  const pred = ekfPredict(app.xEst, app.P, app.u, Q, dt);

  // EKF update
  const upd = ekfUpdate(pred.x, pred.P, meas.y, meas.vis, meas.R);
  app.xEst = upd.x;
  app.P = upd.P;

  // time update
  app.t += dt;
  app.stepCount++;

  // store history
  const H = app.history;
  H.t.push(app.t);
  H.xT.push(app.xTrue.slice());
  H.xE.push(app.xEst.slice());

  // NEES
  let nees = NaN;
  try {
    const e = [
      app.xEst[0] - app.xTrue[0],
      app.xEst[1] - app.xTrue[1],
      app.xEst[2] - app.xTrue[2],
      app.xEst[3] - app.xTrue[3]
    ];
    const Pinv = numeric.inv(app.P);
    const tmp = numeric.dotMV(Pinv, e);
    nees = e[0] * tmp[0] + e[1] * tmp[1] + e[2] * tmp[2] + e[3] * tmp[3];
  } catch {}
  H.nees.push(nees);

  // NIS
  const dof = upd.innov ? upd.innov.length : 0;
  H.nisDof.push(dof);

  let nis = NaN;
  try {
    if (upd.S && upd.innov && dof > 0) {
      const Sinv = numeric.inv(upd.S);
      const tmp2 = numeric.dotMV(Sinv, upd.innov);
      nis = upd.innov.reduce((s, v, i) => s + v * tmp2[i], 0);
    }
  } catch {}
  H.nis.push(nis);

  // store measurement records per-station for plotting
  // (break y vector into per-station triplets)
  if (meas.vis.length) {
    for (let k = 0; k < meas.vis.length; k++) {
      const stationIdx = meas.vis[k].idx;
      const yk = meas.y.slice(3 * k, 3 * k + 3);
      const yhk = upd.yhat.slice(3 * k, 3 * k + 3);
      const ik = upd.innov.slice(3 * k, 3 * k + 3);
      H.meas.push({ t: app.t, stationIdx, y: yk, yhat: yhk, innov: ik, gated: !!upd.gated });
    }
  }

  // crash guard (truth or estimate)
  checkCrash();
}

/* ============================================================
   Readouts + tabs
   ============================================================ */
function updateReadouts() {
  app.el.readoutTime.textContent = `${app.t.toFixed(1)} s`;
  app.el.readoutTimeFine.textContent = `t = ${app.t.toFixed(2)} s`;
  app.el.readoutStep.textContent = `${app.stepCount}`;
  app.el.readoutVisible.textContent = `${app.visibleStations.length}`;
}

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const name = tab.getAttribute("data-tab");
      tabs.forEach(t => {
        t.classList.toggle("is-active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      document.querySelectorAll(".pane").forEach(p => {
        p.classList.toggle("is-active", p.getAttribute("data-pane") === name);
      });
    });
  });
}

/* ============================================================
   Controls
   ============================================================ */
function setRunning(on) {
  app.running = on;
  app.el.statusRun.textContent = on ? "RUNNING" : "PAUSED";
  app.el.statusRun.classList.toggle("pill--idle", !on);
  app.el.btnToggleRun.textContent = on ? "Pause" : "Run";
}

function toggleRun() { setRunning(!app.running); }

function stepSim() {
  simStepFull();
  drawOrbitView();
  drawPlots();
  updateReadouts();
}

function resetView() {
  app.view.centerKm = [0, 0];
  const c = app.orbit.canvas;
  const target = 0.38 * Math.min(c.width, c.height);
  app.view.pxPerKm = target / R_E;
}

function zoomToCraft() {
  app.view.centerKm = [app.xTrue[0], app.xTrue[2]];
}

function resetSim() {
  app.t = 0;
  app.stepCount = 0;
  app.u = [0, 0];
  app.procBias = [0, 0];
  app.xTrue = [7000, 0, 0, 7.5];
  app.xEst = [6999, 0, 0, 7.5];
  app.P = numeric.identity(4);
  app.history = null; // reset history
  ensureHistory();

  // clear plot transforms
  for (const k in app.plots) { app.plots[k].scale = 1; app.plots[k].offsetX = 0; app.plots[k].offsetY = 0; }

  // UI
  app.el.thrAx.textContent = "0.00000";
  app.el.thrAy.textContent = "0.00000";
  setRunning(false);
  resetView();

  drawOrbitView();
  drawPlots();
  updateReadouts();
}

function setupSpeedChips() {
  const chips = Array.from(document.querySelectorAll(".chip[data-speed]"));
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      app.speed = Math.max(1, parseInt(chip.getAttribute("data-speed"), 10) || 1);
    });
  });
}

function setupUI() {
  // cache elements
  const ids = [
    "btnToggleRun","btnStep","btnReset",
    "btnZoomCraft","btnCenterView","btnResetView",
    "inpDt","inpSteps",
    "tglProcNoise","tglMeasNoise",
    "selProcModel","inpProcSigma","inpProcTau",
    "inpRtrueRho","inpRtrueRhod","inpRtruePhi",
    "selCrash","selProc","inpAlpha","inpEps","selGate",
    "btnThUp","btnThDown","btnThLeft","btnThRight","btnThZero","inpThrust",
    "thrAx","thrAy",
    "statusRun","statusTruthDist","truthDistCallout",
    "readoutTime","readoutTimeFine","readoutStep","readoutVisible",
    "orbitCanvas","stationBoard","stationBoardSmall",
    "plotStates","plotErrors","plotMeas","plotNEES"
  ];
  for (const id of ids) app.el[id] = document.getElementById(id);

  // aliases for readability
  app.el.orbitCanvas = app.el.orbitCanvas;
  app.el.stationBoard = app.el.stationBoard;
  app.el.stationBoardSmall = app.el.stationBoardSmall;

  // buttons
  app.el.btnToggleRun.onclick = toggleRun;
  app.el.btnStep.onclick = stepSim;
  app.el.btnReset.onclick = resetSim;

  app.el.btnZoomCraft.onclick = zoomToCraft;
  app.el.btnCenterView.onclick = () => { app.view.centerKm = [0, 0]; };
  app.el.btnResetView.onclick = resetView;

  // noise toggles
  app.el.tglProcNoise.onchange = (e) => {
    app.procNoiseOn = !!e.target.checked;
    app.el.statusTruthDist.hidden = !app.procNoiseOn;
    app.el.truthDistCallout.hidden = !app.procNoiseOn;
  };
  app.el.tglMeasNoise.onchange = (e) => { app.measNoiseOn = !!e.target.checked; };

  // disturbance params
  app.el.selProcModel.onchange = (e) => { app.procModel = e.target.value; };
  app.el.inpProcSigma.onchange = (e) => { app.procSigma = Math.max(0, parseFloat(e.target.value) || 0); };
  app.el.inpProcTau.onchange = (e) => { app.procTau = Math.max(1, parseFloat(e.target.value) || 300); };

  // filter params
  app.el.inpAlpha.onchange = (e) => { app.alpha = clamp(parseFloat(e.target.value) || 0.05, 0.001, 0.2); };
  app.el.inpEps.onchange = (e) => { app.epsFD = Math.max(1e-10, parseFloat(e.target.value) || 1e-6); };
  app.el.selGate.onchange = (e) => { app.gateMode = e.target.value; };
  app.el.selProc.onchange = (e) => { app.procModelEKF = e.target.value; };
  app.el.selCrash.onchange = (e) => { app.crashMode = e.target.value; };

  // thrust controls
  const dirs = {
    btnThUp: [0, 1],
    btnThDown: [0, -1],
    btnThLeft: [-1, 0],
    btnThRight: [1, 0],
    btnThZero: [0, 0]
  };
  const setThrust = (dir) => {
    const mag = Math.max(0, parseFloat(app.el.inpThrust.value) || 0);
    app.u = scale2(dir, mag);
    app.el.thrAx.textContent = app.u[0].toFixed(5);
    app.el.thrAy.textContent = app.u[1].toFixed(5);
  };
  app.el.btnThUp.onclick = () => setThrust(dirs.btnThUp);
  app.el.btnThDown.onclick = () => setThrust(dirs.btnThDown);
  app.el.btnThLeft.onclick = () => setThrust(dirs.btnThLeft);
  app.el.btnThRight.onclick = () => setThrust(dirs.btnThRight);
  app.el.btnThZero.onclick = () => setThrust(dirs.btnThZero);

  // keyboard thrust
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp") setThrust(dirs.btnThUp);
    else if (e.key === "ArrowDown") setThrust(dirs.btnThDown);
    else if (e.key === "ArrowLeft") setThrust(dirs.btnThLeft);
    else if (e.key === "ArrowRight") setThrust(dirs.btnThRight);
    else if (e.key === " ") setThrust(dirs.btnThZero);
  });

  // init values from UI
  app.procNoiseOn = !!app.el.tglProcNoise.checked;
  app.measNoiseOn = !!app.el.tglMeasNoise.checked;
  app.procModel = app.el.selProcModel.value;
  app.procSigma = Math.max(0, parseFloat(app.el.inpProcSigma.value) || app.procSigma);
  app.procTau = Math.max(1, parseFloat(app.el.inpProcTau.value) || app.procTau);
  app.alpha = clamp(parseFloat(app.el.inpAlpha.value) || app.alpha, 0.001, 0.2);
  app.epsFD = Math.max(1e-10, parseFloat(app.el.inpEps.value) || app.epsFD);
  app.gateMode = app.el.selGate.value;
  app.procModelEKF = app.el.selProc.value;
  app.crashMode = app.el.selCrash.value;

  app.el.statusTruthDist.hidden = !app.procNoiseOn;
  app.el.truthDistCallout.hidden = !app.procNoiseOn;

  setupSpeedChips();
}

/* ============================================================
   Main animation loop
   ============================================================ */
function loop() {
  if (app.running) {
    for (let i = 0; i < app.speed; i++) simStepFull();
  }
  drawOrbitView();
  drawPlots();
  updateReadouts();
  requestAnimationFrame(loop);
}

/* ============================================================
   Init
   ============================================================ */
function init() {
  initStations();
  ensureHistory();

  setupUI();
  setupTabs();

  setupOrbitCanvas();
  enablePlotInteractions();

  app.P = numeric.identity(4);

  buildStationBoards();
  resetView();
  setRunning(false);

  drawOrbitView();
  drawPlots();
  updateReadouts();

  requestAnimationFrame(loop);
}

document.addEventListener("DOMContentLoaded", init);
