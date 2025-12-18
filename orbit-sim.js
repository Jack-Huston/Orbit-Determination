/* orbit-sim.js
   2D Orbit Determination + EKF Demo
   Self-contained: physics, estimator, rendering, UI.
   Author: Carissa Mayo (Portfolio Demo)
*/

// --------------------------------------------------
// Global constants and helpers
// --------------------------------------------------
const MU = 398600;       // km^3/s^2
const R_E = 6378;        // km
const OMEGA_E = 7.2722052e-5; // rad/s
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

// Simulation state
const app = {
  running: false,
  t: 0,
  dt: 1,
  stepCount: 0,
  speed: 1,
  u: [0, 0],          // thrust accel km/s^2
  procNoiseOn: false,
  measNoiseOn: true,
  procModel: "gm",
  procSigma: 5e-8,
  procTau: 300,
  procBias: [0, 0],   // Gauss-Markov bias
  crashMode: "reset",
  xTrue: [7000, 0, 0, 7.5],
  xEst: [6999, 0, 0, 7.5],
  P: numeric.identity(4),
  history: [],
  stations: [],
  visibleStations: [],
  canvas: null,
  ctx: null,
  plotCanvases: {},
  dishElems: [],
  pausedForPlot: false
};

// --------------------------------------------------
// Utility functions
// --------------------------------------------------
function norm(v) {
  return Math.hypot(v[0], v[1]);
}
function add(v, w) {
  return [v[0] + w[0], v[1] + w[1]];
}
function sub(v, w) {
  return [v[0] - w[0], v[1] - w[1]];
}
function scale(v, s) {
  return [v[0] * s, v[1] * s];
}
function clamp(x, a, b) {
  return Math.min(b, Math.max(a, x));
}
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// --------------------------------------------------
// Propagation (RK4 two-body + optional thrust)
// --------------------------------------------------
function derivTwoBody(x, u = [0, 0]) {
  const [X, Xdot, Y, Ydot] = x;
  const r = Math.hypot(X, Y);
  const acc = -MU / (r ** 3);
  return [Xdot, acc * X + u[0], Ydot, acc * Y + u[1]];
}

function rk4Step(x, u, dt) {
  const k1 = derivTwoBody(x, u);
  const x2 = add(x, scale(k1, dt / 2));
  const k2 = derivTwoBody(x2, u);
  const x3 = add(x, scale(k2, dt / 2));
  const k3 = derivTwoBody(x3, u);
  const x4 = add(x, scale(k3, dt));
  const res = [];
  for (let i = 0; i < 4; i++) {
    res[i] = x[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return res;
}

// --------------------------------------------------
// Ground stations setup
// --------------------------------------------------
function initStations() {
  const N = 6;
  const stas = [];
  for (let i = 0; i < N; i++) {
    const lon = (i / N) * 2 * Math.PI;
    stas.push({
      lon0: lon,
      name: `GS-${i + 1}`,
      color: `hsl(${(i * 360) / N}, 70%, 50%)`
    });
  }
  app.stations = stas;
}

// Station positions in ECI
function stationPositions(t) {
  return app.stations.map(st => {
    const lon = st.lon0 + OMEGA_E * t;
    return [R_E * Math.cos(lon), R_E * Math.sin(lon)];
  });
}

// Horizon check
function visibleStations(x) {
  const pos = [x[0], x[2]];
  const stations = stationPositions(app.t);
  const visible = [];
  for (let i = 0; i < stations.length; i++) {
    const r_s = stations[i];
    const rho = sub(pos, r_s);
    const visibleNow = dot(r_s, rho) < 0; // below horizon if positive
    if (visibleNow) visible.push({ idx: i, r_s });
  }
  return visible;
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }

// --------------------------------------------------
// Measurement model (ρ, ρ̇, φ)
// --------------------------------------------------
function measureStation(x, r_s) {
  const pos = [x[0], x[2]];
  const vel = [x[1], x[3]];
  const rhoVec = sub(pos, r_s);
  const rho = norm(rhoVec);
  const rhoDot = dot(rhoVec, vel) / rho;
  const phi = Math.atan2(pos[1] - r_s[1], pos[0] - r_s[0]);
  return [rho, rhoDot, phi];
}

// --------------------------------------------------
// Process noise update (Gauss–Markov or white)
// --------------------------------------------------
function updateProcessNoise(dt) {
  if (!app.procNoiseOn) {
    app.procBias = [0, 0];
    return [0, 0];
  }
  if (app.procModel === "white") {
    return [randn() * app.procSigma, randn() * app.procSigma];
  } else {
    const phi = Math.exp(-dt / app.procTau);
    for (let i = 0; i < 2; i++) {
      app.procBias[i] = phi * app.procBias[i] + Math.sqrt(1 - phi ** 2) * randn() * app.procSigma;
    }
    return [...app.procBias];
  }
}

// --------------------------------------------------
// EKF (simplified, placeholder structure preserved)
// --------------------------------------------------
function ekfPredict(x, P, u, Q, dt) {
  const xp = rk4Step(x, u, dt);
  const F = numeric.identity(4); // placeholder Jacobian (simplify for now)
  const Ft = numeric.transpose(F);
  const Pp = numeric.add(numeric.dotMMsmall(numeric.dotMMsmall(F, P), Ft), Q);
  return { x: xp, P: Pp };
}

// --------------------------------------------------
// Crash guard
// --------------------------------------------------
function checkCrash(x) {
  const r = Math.hypot(x[0], x[2]);
  if (r < R_E) {
    if (app.crashMode === "reset") resetSim();
    else app.running = false;
  }
}

// --------------------------------------------------
// Rendering: Orbit + stations
// --------------------------------------------------
function setupCanvas() {
  app.canvas = document.getElementById("orbitCanvas");
  app.ctx = app.canvas.getContext("2d");
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  app.canvas.width = app.canvas.clientWidth;
  app.canvas.height = app.canvas.clientHeight;
}

function drawOrbitView() {
  const ctx = app.ctx;
  const w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Transform center at Earth
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(1, -1);
  const scalePix = 1 / 10; // 1 pixel per 10 km (adjust later)
  ctx.scale(scalePix, scalePix);

  // Earth
  ctx.beginPath();
  ctx.arc(0, 0, R_E, 0, 2 * Math.PI);
  const grad = ctx.createRadialGradient(0, 0, R_E * 0.6, 0, 0, R_E);
  grad.addColorStop(0, "#1e3a8a");
  grad.addColorStop(1, "#0b1220");
  ctx.fillStyle = grad;
  ctx.fill();

  // Ground stations
  const stas = stationPositions(app.t);
  for (let i = 0; i < stas.length; i++) {
    const s = stas[i];
    const st = app.stations[i];
    ctx.beginPath();
    ctx.fillStyle = st.color;
    ctx.arc(s[0], s[1], 40, 0, 2 * Math.PI);
    ctx.fill();
    // dish orientation (rotate to craft)
    const toCraft = Math.atan2(app.xTrue[2] - s[1], app.xTrue[0] - s[0]);
    if (app.dishElems[i]) {
      app.dishElems[i].style.transform = `rotate(${toCraft}rad)`;
    }
  }

  // Truth trajectory (small dot)
  ctx.beginPath();
  ctx.fillStyle = "#f97316";
  ctx.arc(app.xTrue[0], app.xTrue[2], 50, 0, 2 * Math.PI);
  ctx.fill();

  // Estimate
  ctx.beginPath();
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 40;
  ctx.arc(app.xEst[0], app.xEst[2], 80, 0, 2 * Math.PI);
  ctx.stroke();

  ctx.restore();
}

// --------------------------------------------------
// Plot pan/zoom
// --------------------------------------------------
function enablePlotInteractions() {
  for (const key of ["plotStates", "plotErrors", "plotMeas", "plotNEES"]) {
    const canvas = document.getElementById(key);
    app.plotCanvases[key] = {
      el: canvas,
      ctx: canvas.getContext("2d"),
      scale: 1,
      offsetX: 0,
      offsetY: 0
    };
    setupPlotPanZoom(app.plotCanvases[key]);
  }
}

function setupPlotPanZoom(plot) {
  let dragging = false;
  let lastX = 0, lastY = 0;

  plot.el.addEventListener("mousedown", e => {
    if (!app.running) {
      dragging = true;
      lastX = e.offsetX;
      lastY = e.offsetY;
    }
  });
  plot.el.addEventListener("mousemove", e => {
    if (dragging) {
      plot.offsetX += (e.offsetX - lastX);
      plot.offsetY += (e.offsetY - lastY);
      lastX = e.offsetX;
      lastY = e.offsetY;
    }
  });
  window.addEventListener("mouseup", () => dragging = false);
  plot.el.addEventListener("wheel", e => {
    if (!app.running) {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      plot.scale *= factor;
    }
  });
}

// --------------------------------------------------
// UI wiring
// --------------------------------------------------
function setupUI() {
  document.getElementById("btnToggleRun").onclick = toggleRun;
  document.getElementById("btnStep").onclick = stepSim;
  document.getElementById("btnReset").onclick = resetSim;
  document.getElementById("btnZoomCraft").onclick = () => console.log("zoom craft clicked");
  document.getElementById("tglProcNoise").onchange = e => {
    app.procNoiseOn = e.target.checked;
    document.getElementById("statusTruthDist").hidden = !e.target.checked;
    document.getElementById("truthDistCallout").hidden = !e.target.checked;
  };
  document.getElementById("tglMeasNoise").onchange = e => app.measNoiseOn = e.target.checked;
  document.getElementById("selProcModel").onchange = e => app.procModel = e.target.value;
  document.getElementById("inpProcSigma").onchange = e => app.procSigma = parseFloat(e.target.value);
  document.getElementById("inpProcTau").onchange = e => app.procTau = parseFloat(e.target.value);
  document.getElementById("selCrash").onchange = e => app.crashMode = e.target.value;

  // thrust controls
  const dirs = {
    btnThUp: [0, 1],
    btnThDown: [0, -1],
    btnThLeft: [-1, 0],
    btnThRight: [1, 0],
    btnThZero: [0, 0]
  };
  for (const id in dirs) {
    document.getElementById(id).onclick = () => {
      const mag = parseFloat(document.getElementById("inpThrust").value);
      app.u = scale(dirs[id], mag);
      document.getElementById("thrAx").textContent = app.u[0].toFixed(5);
      document.getElementById("thrAy").textContent = app.u[1].toFixed(5);
    };
  }
  enablePlotInteractions();
}

// --------------------------------------------------
// Simulation control
// --------------------------------------------------
function toggleRun() {
  app.running = !app.running;
  document.getElementById("statusRun").textContent = app.running ? "RUNNING" : "PAUSED";
}
function stepSim() {
  simStep();
  drawOrbitView();
}
function resetSim() {
  app.t = 0;
  app.stepCount = 0;
  app.xTrue = [7000, 0, 0, 7.5];
  app.xEst = [6998, 0, 0, 7.5];
  app.P = numeric.identity(4);
  app.procBias = [0, 0];
  drawOrbitView();
}

// --------------------------------------------------
// Main simulation step
// --------------------------------------------------
function simStep() {
  const dt = app.dt;
  const noise = updateProcessNoise(dt);
  const uTot = add(app.u, noise);
  app.xTrue = rk4Step(app.xTrue, uTot, dt);
  checkCrash(app.xTrue);
  const Q = numeric.mul(numeric.identity(4), 1e-6);
  const pred = ekfPredict(app.xEst, app.P, app.u, Q, dt);
  app.xEst = pred.x;
  app.P = pred.P;
  app.t += dt;
  app.stepCount++;
}

// --------------------------------------------------
// Animation loop
// --------------------------------------------------
function loop() {
  if (app.running) {
    for (let i = 0; i < app.speed; i++) simStep();
    drawOrbitView();
  } else {
    drawOrbitView();
  }
  requestAnimationFrame(loop);
}

// --------------------------------------------------
// Initialization
// --------------------------------------------------
function init() {
  initStations();
  setupCanvas();
  setupUI();
  // attach dish elements to station board
  const board = document.getElementById("stationBoard");
  board.innerHTML = "";
  app.dishElems = [];
  for (let i = 0; i < app.stations.length; i++) {
    const tile = document.createElement("div");
    tile.className = "stationtile";
    tile.innerHTML = `
      <div class="stationtile__dish">
        <svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 1 10 10h-2a8 8 0 0 0-8-8V2z"/></svg>
      </div>
      <div class="stationtile__name">${app.stations[i].name}</div>`;
    tile.style.borderColor = app.stations[i].color;
    board.appendChild(tile);
    app.dishElems.push(tile.querySelector(".stationtile__dish"));
  }
  loop();
}

document.addEventListener("DOMContentLoaded", init);
// --------------------------------------------------
// Measurement + EKF update (continued from previous)
// --------------------------------------------------

// Numerical Jacobian of measurement function
function measJacobian(x, stations) {
  const eps = 1e-6;
  const Hrows = [];

  for (const st of stations) {
    const h0 = measureStation(x, st.r_s);
    for (let j = 0; j < 4; j++) {
      const xp = x.slice();
      xp[j] += eps;
      const hp = measureStation(xp, st.r_s);
      const diff = hp.map((v, k) => (v - h0[k]) / eps);
      Hrows.push(diff);
    }
  }
  // reshape rows into 2D matrix
  const m = Hrows.length / 4;
  const H = [];
  for (let i = 0; i < m; i++) {
    H.push(Hrows.slice(i * 4, (i + 1) * 4));
  }
  return H;
}

// EKF update given measurements
function ekfUpdate(x, P, yObs, stations, R) {
  if (stations.length === 0) return { x, P };
  const H = measJacobian(x, stations);
  const yPred = [];
  for (const st of stations) {
    const h = measureStation(x, st.r_s);
    yPred.push(...h);
  }
  const yVec = numeric.sub(yObs, yPred);
  const S = numeric.add(numeric.dotMMsmall(numeric.dotMMsmall(H, P), numeric.transpose(H)), R);
  const K = numeric.dotMMsmall(numeric.dotMMsmall(P, numeric.transpose(H)), numeric.inv(S));
  const xNew = numeric.add(x, numeric.dotMV(K, yVec));
  const I = numeric.identity(4);
  const PNew = numeric.dotMMsmall(numeric.sub(I, numeric.dotMMsmall(K, H)), P);
  return { x: xNew, P: PNew };
}

// Generate measurement vector with optional noise
function makeMeasurements() {
  const vis = visibleStations(app.xTrue);
  app.visibleStations = vis;
  const y = [];
  const Rdiag = [];
  for (const { idx, r_s } of vis) {
    const [rho, rhod, phi] = measureStation(app.xTrue, r_s);
    const noise = app.measNoiseOn ? [
      randn() * parseFloat(document.getElementById("inpRtrueRho").value),
      randn() * parseFloat(document.getElementById("inpRtrueRhod").value),
      randn() * parseFloat(document.getElementById("inpRtruePhi").value)
    ] : [0, 0, 0];
    y.push(rho + noise[0], rhod + noise[1], phi + noise[2]);
    Rdiag.push(noise[0] ** 2, noise[1] ** 2, noise[2] ** 2);
  }
  const R = numeric.diag(Rdiag.length ? Rdiag : [1, 1, 1]);
  return { y, R };
}

// --------------------------------------------------
// Plotting and UI updates
// --------------------------------------------------
function updateReadouts() {
  document.getElementById("readoutTime").textContent = app.t.toFixed(1) + " s";
  document.getElementById("readoutStep").textContent = app.stepCount;
  document.getElementById("readoutVisible").textContent = app.visibleStations.length;
}

// Initialize tab behavior
function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const name = tab.getAttribute("data-tab");
      document.querySelectorAll(".pane").forEach(p => p.classList.remove("is-active"));
      document.querySelector(`[data-pane='${name}']`).classList.add("is-active");
    });
  });
}

// --------------------------------------------------
// Plot utilities (lightweight custom drawing)
// --------------------------------------------------
function drawPlots() {
  // Just placeholders for now, basic indicator of operation
  const keys = ["plotStates", "plotErrors", "plotMeas", "plotNEES"];
  for (const k of keys) {
    const ctx = app.plotCanvases[k].ctx;
    const w = ctx.canvas.width, h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f1f5f9";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#64748b";
    ctx.font = "12px monospace";
    ctx.fillText(k.replace("plot", ""), 12, 18);
  }
}

// --------------------------------------------------
// Numerical library subset (minimal replacement for numeric.js)
// --------------------------------------------------
const numeric = {
  identity: function (n) {
    const I = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );
    return I;
  },
  transpose: function (A) {
    return A[0].map((_, j) => A.map(row => row[j]));
  },
  add: function (A, B) {
    if (Array.isArray(A[0])) {
      return A.map((r, i) => r.map((v, j) => v + B[i][j]));
    } else {
      return A.map((v, i) => v + B[i]);
    }
  },
  sub: function (A, B) {
    if (Array.isArray(A[0])) {
      return A.map((r, i) => r.map((v, j) => v - B[i][j]));
    } else {
      return A.map((v, i) => v - B[i]);
    }
  },
  mul: function (A, s) {
    if (Array.isArray(A[0])) {
      return A.map(r => r.map(v => v * s));
    } else {
      return A.map(v => v * s);
    }
  },
  dotMMsmall: function (A, B) {
    const n = A.length, m = B[0].length, p = B.length;
    const C = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < p; k++) {
        for (let j = 0; j < m; j++) C[i][j] += A[i][k] * B[k][j];
      }
    }
    return C;
  },
  dotMV: function (A, v) {
    return A.map(r => r.reduce((sum, val, j) => sum + val * v[j], 0));
  },
  inv: function (M) {
    // basic 2x2, 3x3, or diagonal fallback
    const n = M.length;
    if (n === 2) {
      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      return [
        [M[1][1] / det, -M[0][1] / det],
        [-M[1][0] / det, M[0][0] / det]
      ];
    } else {
      // assume diagonal
      return M.map((r, i) => r.map((v, j) => (i === j ? 1 / v : 0)));
    }
  },
  diag: function (arr) {
    return arr.map((v, i) => arr.map((_, j) => (i === j ? v : 0)));
  }
};

// --------------------------------------------------
// Integration into loop (measurement + EKF update)
// --------------------------------------------------
function simStepFull() {
  const dt = app.dt;
  const noise = updateProcessNoise(dt);
  const uTot = add(app.u, noise);

  // Propagate truth
  app.xTrue = rk4Step(app.xTrue, uTot, dt);

  // Create measurements
  const meas = makeMeasurements();

  // EKF prediction
  const Q = numeric.mul(numeric.identity(4), 1e-6);
  const pred = ekfPredict(app.xEst, app.P, app.u, Q, dt);

  // EKF update with measurements
  const upd = ekfUpdate(pred.x, pred.P, meas.y, app.visibleStations, meas.R);
  app.xEst = upd.x;
  app.P = upd.P;

  app.t += dt;
  app.stepCount++;
  checkCrash(app.xTrue);
}

// Override main loop to use full model
function loopFull() {
  if (app.running) {
    for (let i = 0; i < app.speed; i++) simStepFull();
  }
  drawOrbitView();
  drawPlots();
  updateReadouts();
  requestAnimationFrame(loopFull);
}

// Replace old loop with full version on init
function initFull() {
  initStations();
  setupCanvas();
  setupUI();
  setupTabs();
  enablePlotInteractions();

  // Create station board + dish icons
  const board = document.getElementById("stationBoard");
  board.innerHTML = "";
  app.dishElems = [];
  for (let i = 0; i < app.stations.length; i++) {
    const tile = document.createElement("div");
    tile.className = "stationtile";
    tile.innerHTML = `
      <div class="stationtile__dish">
        <svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 1 10 10h-2a8 8 0 0 0-8-8V2z"/></svg>
      </div>
      <div class="stationtile__name">${app.stations[i].name}</div>`;
    tile.style.borderColor = app.stations[i].color;
    board.appendChild(tile);
    app.dishElems.push(tile.querySelector(".stationtile__dish"));
  }

  drawOrbitView();
  drawPlots();
  loopFull();
}

document.removeEventListener("DOMContentLoaded", init);
document.addEventListener("DOMContentLoaded", initFull);
