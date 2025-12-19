/* orbit-sim.js
   2D Orbit Determination + EKF + Orbit Control (client-side, no libraries)
   Units: km, km/s, rad, s. Earth constants fixed.

   Controller:
   - Reference orbit defined by (perigee altitude, eccentricity, arg. periapsis)
   - Two control modes:
       1) Nonlinear tracking PI-D (default): u = a_ref - a_hat + Kp e_r + Kd e_v + Ki ∫e_r dt
       2) Linear LQR (near-circular): LQR on [δr, δṙ, δθ, δθ̇] in LVLH; mapped to inertial accel.
   - Disturbance: optional truth acceleration disturbance (white or Gauss–Markov)
*/

(() => {
  "use strict";

  // -----------------------------
  // Constants
  // -----------------------------
  const MU = 398600;            // km^3/s^2
  const R_E = 6378;             // km
  const OMEGA_E = 7.2722052e-5; // rad/s
  const DEG2RAD = Math.PI / 180;
  const RAD2DEG = 180 / Math.PI;
  const G0 = 9.80665;           // m/s^2
  const MS2_TO_KMPS2 = 1e-3;    // (m/s^2) -> (km/s^2)

  const ST_COLORS = [
    "#ef4444","#f97316","#f59e0b","#84cc16",
    "#22c55e","#14b8a6","#06b6d4","#0ea5e9",
    "#3b82f6","#6366f1","#8b5cf6","#ec4899"
  ];

  // -----------------------------
  // DOM
  // -----------------------------
  const el = {
    orbitCanvas: document.getElementById("orbitCanvas"),
    plotStates: document.getElementById("plotStates"),
    plotErrors: document.getElementById("plotErrors"),
    plotMeas: document.getElementById("plotMeas"),
    plotTraj: document.getElementById("plotTraj"),
    plotCons: document.getElementById("plotCons"),
    plotCtrl: document.getElementById("plotCtrl"),
    stationBoard: document.getElementById("stationBoard"),

    btnToggleRun: document.getElementById("btnToggleRun"),
    btnStep: document.getElementById("btnStep"),
    btnReset: document.getElementById("btnReset"),
    btnZoomCraft: document.getElementById("btnZoomCraft"),
    btnResetView: document.getElementById("btnResetView"),
    btnApplyTarget: document.getElementById("btnApplyTarget"),
    btnHoldHere: document.getElementById("btnHoldHere"),

    inpDt: document.getElementById("inpDt"),
    inpMeasEvery: document.getElementById("inpMeasEvery"),

    inpTgtAlt: document.getElementById("inpTgtAlt"),
    inpTgtE: document.getElementById("inpTgtE"),
    inpTgtW: document.getElementById("inpTgtW"),
    selPhase: document.getElementById("selPhase"),

    tglCtrl: document.getElementById("tglCtrl"),
    tglCtrlUseEKF: document.getElementById("tglCtrlUseEKF"),
    selCtrlMode: document.getElementById("selCtrlMode"),
    inpUMaxG: document.getElementById("inpUMaxG"),
    inpAgg: document.getElementById("inpAgg"),
    inpKp: document.getElementById("inpKp"),
    inpKd: document.getElementById("inpKd"),
    inpKi: document.getElementById("inpKi"),
    inpIClamp: document.getElementById("inpIClamp"),
    calloutCtrlSat: document.getElementById("calloutCtrlSat"),

    tglTruthDist: document.getElementById("tglTruthDist"),
    selDistModel: document.getElementById("selDistModel"),
    inpDistSigma: document.getElementById("inpDistSigma"),
    inpDistTau: document.getElementById("inpDistTau"),
    pillDist: document.getElementById("pillDist"),
    overlayWarn: document.getElementById("overlayWarn"),

    tglMeasNoise: document.getElementById("tglMeasNoise"),
    inpSigRho: document.getElementById("inpSigRho"),
    inpSigRhod: document.getElementById("inpSigRhod"),
    inpSigPhi: document.getElementById("inpSigPhi"),
    inpQsigma: document.getElementById("inpQsigma"),
    inpAlpha: document.getElementById("inpAlpha"),

    readoutRun: document.getElementById("readoutRun"),
    readoutTime: document.getElementById("readoutTime"),
    readoutDt: document.getElementById("readoutDt"),
    readoutVisible: document.getElementById("readoutVisible"),
    readoutDV: document.getElementById("readoutDV"),
    readoutOrbit: document.getElementById("readoutOrbit"),
    readoutA: document.getElementById("readoutA"),
    readoutRa: document.getElementById("readoutRa"),
    readoutPeriod: document.getElementById("readoutPeriod"),
  };

  // -----------------------------
  // Math helpers
  // -----------------------------
  const clamp = (x,a,b)=>Math.min(b,Math.max(a,x));
  const hypot2 = (x,y)=>Math.hypot(x,y);
  const dot2 = (ax,ay,bx,by)=>ax*bx+ay*by;
  const wrapPi = (a)=>Math.atan2(Math.sin(a),Math.cos(a));
  const fmt = (x, d=3)=> (Number.isFinite(x)? x.toFixed(d): "—");

  function randn() {
    // Box-Muller
    let u=0,v=0;
    while(u===0) u=Math.random();
    while(v===0) v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  }

  // -----------------------------
  // Linear algebra (small matrices)
  // -----------------------------
  const zeros = (r,c)=>Array.from({length:r}, ()=>Array(c).fill(0));
  const eye = (n)=>Array.from({length:n}, (_,i)=>Array.from({length:n}, (_,j)=>i===j?1:0));

  function matTrans(A){
    const r=A.length,c=A[0].length, AT=zeros(c,r);
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) AT[j][i]=A[i][j];
    return AT;
  }
  function matMul(A,B){
    const r=A.length,k=A[0].length,c=B[0].length,C=zeros(r,c);
    for(let i=0;i<r;i++){
      for(let j=0;j<c;j++){
        let s=0;
        for(let t=0;t<k;t++) s+=A[i][t]*B[t][j];
        C[i][j]=s;
      }
    }
    return C;
  }
  function matAdd(A,B){
    const r=A.length,c=A[0].length,C=zeros(r,c);
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) C[i][j]=A[i][j]+B[i][j];
    return C;
  }
  function matSub(A,B){
    const r=A.length,c=A[0].length,C=zeros(r,c);
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) C[i][j]=A[i][j]-B[i][j];
    return C;
  }
  function matScale(A,s){
    const r=A.length,c=A[0].length,B=zeros(r,c);
    for(let i=0;i<r;i++) for(let j=0;j<c;j++) B[i][j]=A[i][j]*s;
    return B;
  }
  function matMulVec(A,x){
    const r=A.length,c=A[0].length,y=new Float64Array(r);
    for(let i=0;i<r;i++){
      let s=0;
      for(let j=0;j<c;j++) s+=A[i][j]*x[j];
      y[i]=s;
    }
    return y;
  }
  function matInv(A){
    const n=A.length;
    const M=zeros(n,2*n);
    for(let i=0;i<n;i++){
      for(let j=0;j<n;j++) M[i][j]=A[i][j];
      M[i][n+i]=1;
    }
    for(let col=0;col<n;col++){
      let piv=col, best=Math.abs(M[col][col]);
      for(let r=col+1;r<n;r++){
        const v=Math.abs(M[r][col]);
        if(v>best){best=v;piv=r;}
      }
      if(best<1e-18) throw new Error("Singular matrix.");
      if(piv!==col){ const tmp=M[col]; M[col]=M[piv]; M[piv]=tmp; }
      const p=M[col][col];
      for(let j=0;j<2*n;j++) M[col][j]/=p;
      for(let r=0;r<n;r++){
        if(r===col) continue;
        const f=M[r][col];
        if(f===0) continue;
        for(let j=0;j<2*n;j++) M[r][j]-=f*M[col][j];
      }
    }
    const Inv=zeros(n,n);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) Inv[i][j]=M[i][n+j];
    return Inv;
  }
  function symmetrize(A){
    const n=A.length,B=zeros(n,n);
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) B[i][j]=0.5*(A[i][j]+A[j][i]);
    return B;
  }

  // stats for NEES/NIS bounds
  function normInv(p){
    const a=[-39.69683028665376,220.9460984245205,-275.9285104469687,138.357751867269, -30.66479806614716,2.506628277459239];
    const b=[-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572];
    const c=[-0.007784894002430293,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
    const d=[0.007784695709041462,0.3224671290700398,2.445134137142996,3.754408661907416];
    const plow=0.02425, phigh=1-plow;
    if(p<=0) return -Infinity;
    if(p>=1) return Infinity;
    let q,r;
    if(p<plow){
      q=Math.sqrt(-2*Math.log(p));
      return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    if(p>phigh){
      q=Math.sqrt(-2*Math.log(1-p));
      return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    q=p-0.5; r=q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }
  function chi2Inv(p,k){
    const z=normInv(p);
    const a=2/(9*k);
    const q=k*Math.pow(1-a+z*Math.sqrt(a),3);
    return Math.max(0,q);
  }

  // -----------------------------
  // Orbital conversions
  // -----------------------------
  function accelGrav(x,y){
    const r=hypot2(x,y);
    const s=-MU/(r*r*r);
    return [s*x, s*y];
  }

  function oeFromState(X,VX,Y,VY){
    const r=hypot2(X,Y);
    const v2=VX*VX+VY*VY;
    const eps=v2/2 - MU/r;
    const a = (-MU)/(2*eps);
    const rv = X*VX + Y*VY;
    // evec = (1/mu)*((v^2 - mu/r) r - (r·v) v)
    const c1 = (v2 - MU/r);
    const ex = (c1*X - rv*VX)/MU;
    const ey = (c1*Y - rv*VY)/MU;
    const e = hypot2(ex,ey);
    return {a,e,ex,ey};
  }

  function stateFromPerigeeAltE(altKm,e,omegaDeg,phaseMode,thetaNow){
    const rp = R_E + altKm;
    const ee = clamp(e, 0, 0.95);
    const a = rp/(1-ee);
    const p = a*(1-ee*ee);
    const omega = omegaDeg*DEG2RAD;

    let nu;
    if(phaseMode==="perigee"){
      nu = 0;
    } else {
      // match inertial angle thetaNow (theta = omega + nu)
      nu = wrapPi(thetaNow - omega);
    }

    const r = p/(1 + ee*Math.cos(nu));
    const x_p = r*Math.cos(nu);
    const y_p = r*Math.sin(nu);
    const vfac = Math.sqrt(MU/p);
    const vx_p = -vfac*Math.sin(nu);
    const vy_p =  vfac*(ee + Math.cos(nu));

    // rotate by omega
    const c=Math.cos(omega), s=Math.sin(omega);
    const X = c*x_p - s*y_p;
    const Y = s*x_p + c*y_p;
    const VX = c*vx_p - s*vy_p;
    const VY = s*vx_p + c*vy_p;

    const ra = a*(1+ee);
    const T = 2*Math.PI*Math.sqrt(a*a*a/MU);

    return {X,VX,Y,VY,a,e:ee,rp,ra,T,omega,nu};
  }

  // -----------------------------
  // Dynamics (RK4)
  // state x = [X,VX,Y,VY]
  // -----------------------------
  function fDyn(x,u){
    const X=x[0], VX=x[1], Y=x[2], VY=x[3];
    const [axg, ayg] = accelGrav(X,Y);
    return [VX, axg + u[0], VY, ayg + u[1]];
  }
  function rk4(x,u,dt){
    const k1=fDyn(x,u);
    const x2=[x[0]+0.5*dt*k1[0], x[1]+0.5*dt*k1[1], x[2]+0.5*dt*k1[2], x[3]+0.5*dt*k1[3]];
    const k2=fDyn(x2,u);
    const x3=[x[0]+0.5*dt*k2[0], x[1]+0.5*dt*k2[1], x[2]+0.5*dt*k2[2], x[3]+0.5*dt*k2[3]];
    const k3=fDyn(x3,u);
    const x4=[x[0]+dt*k3[0], x[1]+dt*k3[1], x[2]+dt*k3[2], x[3]+dt*k3[3]];
    const k4=fDyn(x4,u);
    return [
      x[0] + dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,
      x[1] + dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
      x[2] + dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6,
      x[3] + dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6,
    ];
  }
  function A_jac(x){
    const X=x[0], Y=x[2];
    const r=hypot2(X,Y);
    const r5 = Math.pow(r,5);
    const r3 = r*r*r;
    const mu=MU;
    const dadr = -mu/r3; // unused
    // d(ax)/dX = -mu*(1/r^3 - 3X^2/r^5)
    const a11 = -mu*(1/r3 - 3*X*X/r5);
    const a13 =  3*mu*X*Y/r5;
    const a31 = a13;
    const a33 = -mu*(1/r3 - 3*Y*Y/r5);
    return [
      [0,1,0,0],
      [a11,0,a13,0],
      [0,0,0,1],
      [a31,0,a33,0],
    ];
  }

  // -----------------------------
  // Stations & measurements
  // -----------------------------
  function stationState(i,t){
    const lon0 = (i/12)*2*Math.PI;
    const th = lon0 + OMEGA_E*t;
    const X = R_E*Math.cos(th);
    const Y = R_E*Math.sin(th);
    const VX = -R_E*OMEGA_E*Math.sin(th);
    const VY =  R_E*OMEGA_E*Math.cos(th);
    return {X,Y,VX,VY,th};
  }
  function visibleStations(x,t){
    const X=x[0], Y=x[2];
    const vis=[];
    for(let i=0;i<12;i++){
      const s=stationState(i,t);
      const rx=X-s.X, ry=Y-s.Y;
      // visible if LOS is above tangent plane: dot(r_s, rho) > 0
      if(dot2(s.X,s.Y, rx,ry) > 0) vis.push(i+1);
    }
    return vis;
  }
  function hMeas(x, s){
    const X=x[0], VX=x[1], Y=x[2], VY=x[3];
    const rx=X-s.X, ry=Y-s.Y;
    const r = hypot2(rx,ry);
    const rvx=VX-s.VX, rvy=VY-s.VY;
    const rhod = (rx*rvx + ry*rvy)/Math.max(1e-9,r);
    const phi = Math.atan2(ry, rx);
    return [r, rhod, phi];
  }
  function H_meas_num(x, ids){
    const eps=1e-6;
    const m=ids.length*3;
    const H=zeros(m,4);
    if(m===0) return H;
    // baseline measurement vector
    const y0=new Float64Array(m);
    for(let k=0;k<ids.length;k++){
      const s=stationState(ids[k]-1, app.t);
      const h=hMeas(x,s);
      y0[3*k]=h[0]; y0[3*k+1]=h[1]; y0[3*k+2]=h[2];
    }
    for(let j=0;j<4;j++){
      const xp=x.slice();
      xp[j]+=eps;
      for(let k=0;k<ids.length;k++){
        const s=stationState(ids[k]-1, app.t);
        const hp=hMeas(xp,s);
        H[3*k][j]   = (hp[0]-y0[3*k])/eps;
        H[3*k+1][j] = (hp[1]-y0[3*k+1])/eps;
        // angle wrap-safe diff
        const dphi = wrapPi(hp[2]-y0[3*k+2]);
        H[3*k+2][j] = dphi/eps;
      }
    }
    return H;
  }

  // -----------------------------
  // EKF
  // -----------------------------
  class EKF {
    constructor(x0,P0){
      this.x = x0.slice();
      this.P = P0;
      this.lastInnov = null;
      this.lastS = null;
      this.lastDf = 0;
    }

    predict(dt, u, qSigma_ms2){
      // propagate with RK4 (same model as truth)
      this.x = rk4(this.x, u, dt);

      // linearize & discretize with Euler
      const A=A_jac(this.x);
      const F=matAdd(eye(4), matScale(A, dt));

      // accel noise -> state covariance
      const sig = qSigma_ms2*MS2_TO_KMPS2;
      const q = sig*sig;
      // G maps accel to state: [0;1;0;0] for ax, [0;0;0;1] for ay
      const Q = [
        [0,0,0,0],
        [0,q*dt,0,0],
        [0,0,0,0],
        [0,0,0,q*dt],
      ];

      const FP = matMul(F, this.P);
      const Pp = matMul(FP, matTrans(F));
      this.P = symmetrize(matAdd(Pp, Q));
    }

    update(y, ids, Rdiag){
      const m=ids.length*3;
      this.lastDf = m;
      if(m===0) return {nis: NaN};

      const H = H_meas_num(this.x, ids);

      // yhat
      const yhat=new Float64Array(m);
      for(let k=0;k<ids.length;k++){
        const s=stationState(ids[k]-1, app.t);
        const h=hMeas(this.x,s);
        yhat[3*k]=h[0]; yhat[3*k+1]=h[1]; yhat[3*k+2]=h[2];
      }

      const innov=new Float64Array(m);
      for(let i=0;i<m;i++) innov[i]=y[i]-yhat[i];
      // wrap angles
      for(let k=0;k<ids.length;k++){
        innov[3*k+2]=wrapPi(innov[3*k+2]);
      }

      // S = HPH' + R
      const HP = matMul(H, this.P);
      let S = matMul(HP, matTrans(H));
      for(let i=0;i<m;i++) S[i][i] += Rdiag[i];

      let Sinv;
      try { Sinv = matInv(S); } catch(e){ return {nis: NaN}; }

      // K = P H' S^-1
      const PHt = matMul(this.P, matTrans(H));
      const K = matMul(PHt, Sinv);

      // state update x = x + K*innov
      const dx = matMulVec(K, innov);
      for(let i=0;i<4;i++) this.x[i] += dx[i];

      // Joseph form for numerical stability
      const I = eye(4);
      const KH = matMul(K, H);
      const IKH = matSub(I, KH);
      const IKH_P = matMul(IKH, this.P);
      const IKH_P_IKHt = matMul(IKH_P, matTrans(IKH));
      const KRKt = (() => {
        // K R K'
        const R = zeros(m,m);
        for(let i=0;i<m;i++) R[i][i]=Rdiag[i];
        return matMul(matMul(K, R), matTrans(K));
      })();
      this.P = symmetrize(matAdd(IKH_P_IKHt, KRKt));

      // NIS = innov' S^-1 innov
      const tmp = matMulVec(Sinv, innov);
      let nis=0;
      for(let i=0;i<m;i++) nis += innov[i]*tmp[i];

      this.lastInnov = innov;
      this.lastS = S;

      return {nis};
    }
  }

  // -----------------------------
  // Disturbance model
  // -----------------------------
  class Disturbance {
    constructor(){
      this.bias = [0,0];
    }
    step(dt, enabled, model, sigma_ms2, tau_s){
      if(!enabled){
        this.bias[0]=0; this.bias[1]=0;
        return [0,0];
      }
      const sig = sigma_ms2*MS2_TO_KMPS2;
      if(model==="white"){
        return [randn()*sig, randn()*sig];
      }
      // Gauss-Markov
      const phi = Math.exp(-dt/Math.max(1e-6,tau_s));
      const s = Math.sqrt(Math.max(0,1-phi*phi))*sig;
      this.bias[0] = phi*this.bias[0] + s*randn();
      this.bias[1] = phi*this.bias[1] + s*randn();
      return [this.bias[0], this.bias[1]];
    }
  }

  // -----------------------------
  // Controller
  // -----------------------------
  class Controller {
    constructor(){
      this.eInt = [0,0];
      this.lastU = [0,0];
      this.saturated = false;
      this.lqrK = null;
      this.lqrR0 = null;
    }

    reset(){
      this.eInt=[0,0];
      this.lastU=[0,0];
      this.saturated=false;
      this.lqrK=null;
      this.lqrR0=null;
    }

    compute(dt, mode, enabled, useEKF, xHat, xRef, uMax_kmps2, gains){
      this.saturated=false;
      if(!enabled) return [0,0];

      if(mode==="lqr"){
        return this.computeLQR(dt, xHat, xRef, uMax_kmps2, gains);
      }
      return this.computeTrack(dt, xHat, xRef, uMax_kmps2, gains);
    }

    computeTrack(dt, xHat, xRef, uMax, gains){
      const Kp=gains.Kp, Kd=gains.Kd, Ki=gains.Ki;
      const iclamp=gains.iClamp;

      const erx = xRef[0]-xHat[0];
      const ery = xRef[2]-xHat[2];
      const evx = xRef[1]-xHat[1];
      const evy = xRef[3]-xHat[3];

      // integrate with clamp on each component (km·s)
      this.eInt[0] = clamp(this.eInt[0] + erx*dt, -iclamp, iclamp);
      this.eInt[1] = clamp(this.eInt[1] + ery*dt, -iclamp, iclamp);

      const aRef = accelGrav(xRef[0], xRef[2]);
      const aHat = accelGrav(xHat[0], xHat[2]);

      let ux = (aRef[0]-aHat[0]) + Kp*erx + Kd*evx + Ki*this.eInt[0];
      let uy = (aRef[1]-aHat[1]) + Kp*ery + Kd*evy + Ki*this.eInt[1];

      // soft rate limit
      const rate = 5*uMax;
      ux = clamp(ux, this.lastU[0]-rate*dt, this.lastU[0]+rate*dt);
      uy = clamp(uy, this.lastU[1]-rate*dt, this.lastU[1]+rate*dt);

      // saturate
      const mag = hypot2(ux,uy);
      if(mag>uMax && mag>0){
        ux *= uMax/mag; uy *= uMax/mag;
        this.saturated=true;
      }
      this.lastU[0]=ux; this.lastU[1]=uy;
      return [ux,uy];
    }

    // Linear LQR around near-circular orbit: x_lin=[δr, δṙ, δθ, δθ̇], u=[u_r, u_t]
    computeLQR(dt, xHat, xRef, uMax, gains){
      const r0 = hypot2(xRef[0], xRef[2]);
      const n = Math.sqrt(MU/(r0*r0*r0));

      // Recompute K if reference radius changed meaningfully
      if(!this.lqrK || !this.lqrR0 || Math.abs(r0-this.lqrR0) > 5){
        this.lqrK = lqrGainDiscrete(r0, dt, gains.lqrAgg, uMax);
        this.lqrR0 = r0;
      }
      const K = this.lqrK; // 2x4

      // build linear state from estimate vs reference
      const thHat = Math.atan2(xHat[2], xHat[0]);
      const thRef = Math.atan2(xRef[2], xRef[0]);
      const rHat = hypot2(xHat[0], xHat[2]);
      const er = rHat - r0;

      // polar rates
      const rdotHat = (xHat[0]*xHat[1] + xHat[2]*xHat[3]) / Math.max(1e-9,rHat);
      const hHat = xHat[0]*xHat[3] - xHat[2]*xHat[1];
      const thdotHat = hHat / Math.max(1e-9, rHat*rHat);

      const dth = wrapPi(thHat - thRef);
      const dthdot = thdotHat - n;

      const xlin = new Float64Array([er, rdotHat, dth, dthdot]);

      // u_r/u_t = -K xlin
      const ur = -(K[0][0]*xlin[0] + K[0][1]*xlin[1] + K[0][2]*xlin[2] + K[0][3]*xlin[3]);
      const ut = -(K[1][0]*xlin[0] + K[1][1]*xlin[1] + K[1][2]*xlin[2] + K[1][3]*xlin[3]);

      // map to inertial
      const cr = Math.cos(thRef), sr = Math.sin(thRef);
      const erx = cr, ery = sr;
      const etx = -sr, ety = cr;
      let ux = ur*erx + ut*etx;
      let uy = ur*ery + ut*ety;

      // saturate
      const mag=hypot2(ux,uy);
      if(mag>uMax && mag>0){
        ux*=uMax/mag; uy*=uMax/mag;
        this.saturated=true;
      }
      this.lastU[0]=ux; this.lastU[1]=uy;
      return [ux,uy];
    }
  }

  function lqrGainDiscrete(r0, dt, agg, uMax){
    // Linearized continuous A,B at radius r0 (from user's MATLAB)
    const n = Math.sqrt(MU/(r0*r0*r0));
    const A = [
      [0, 1, 0, 0],
      [3*n*n, 0, 0, 2*n*r0],
      [0, 0, 0, 1],
      [0, -2*n/(r0), 0, 0],
    ];
    const B = [
      [0, 0],
      [1, 0],
      [0, 0],
      [0, 1/r0],
    ];

    // Discretize (Euler; enough for demo)
    const Ad = matAdd(eye(4), matScale(A, dt));
    const Bd = matScale(B, dt);

    // Q/R via simple Bryson-ish default, scaled by aggressiveness
    // state max: 1 km, 0.01 km/s, 0.01 rad, 1e-4 rad/s
    const xmax=[1, 0.01, 0.01, 1e-4];
    const Q = zeros(4,4);
    const w = [0.25, 0.10, 0.55, 0.10]; // weight fractions
    for(let i=0;i<4;i++) Q[i][i] = agg*(w[i]/(xmax[i]*xmax[i]));
    // input max accel in km/s^2
    const umax = Math.max(1e-12, uMax);
    const R = [
      [ (1/ (umax*umax)) / agg, 0 ],
      [ 0, (1/ (umax*umax)) / agg ],
    ];

    // Riccati iteration
    let P = Q;
    const AdT = matTrans(Ad);
    const BdT = matTrans(Bd);

    for(let it=0; it<200; it++){
      const BtP = matMul(BdT, P);
      const BtPB = matMul(BtP, Bd);
      const S = matAdd(R, BtPB);
      const Sinv = matInv(S);
      const AtP = matMul(AdT, P);
      const AtPA = matMul(AtP, Ad);
      const AtPB = matMul(AtP, Bd);
      const mid = matMul(matMul(AtPB, Sinv), matTrans(AtPB));
      const Pn = matAdd(Q, matSub(AtPA, mid));

      // convergence check
      let maxd=0;
      for(let i=0;i<4;i++) maxd=Math.max(maxd, Math.abs(Pn[i][i]-P[i][i]));
      P = Pn;
      if(maxd<1e-10) break;
    }

    // K = (R + B'PB)^-1 (B' P A)
    const BtP = matMul(BdT, P);
    const BtPB = matMul(BtP, Bd);
    const S = matAdd(R, BtPB);
    const Sinv = matInv(S);
    const BtPA = matMul(BtP, Ad);
    const K = matMul(Sinv, BtPA); // 2x4
    return K;
  }

  // -----------------------------
  // Plot helpers
  // -----------------------------
  function setupHiDPICanvas(canvas){
    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width*dpr));
    const h = Math.max(1, Math.floor(rect.height*dpr));
    if(canvas.width!==w || canvas.height!==h){
      canvas.width=w; canvas.height=h;
    }
    const ctx=canvas.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0); // draw in CSS pixels
    return ctx;
  }
  function clearPanel(ctx,W,H){
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle="#ffffff";
    ctx.fillRect(0,0,W,H);
  }
  function computeMinMax(seriesArr){
    let mn=Infinity,mx=-Infinity;
    for(const s of seriesArr){
      for(let i=0;i<s.length;i++){
        const v=s[i];
        if(!Number.isFinite(v)) continue;
        if(v<mn) mn=v;
        if(v>mx) mx=v;
      }
    }
    if(!Number.isFinite(mn) || !Number.isFinite(mx)) return {mn:0,mx:1};
    if(mx-mn<1e-9){ mx=mn+1; }
    // pad
    const pad=0.06*(mx-mn);
    return {mn:mn-pad,mx:mx+pad};
  }
  function drawAxes(ctx,x,y,w,h,opt){
    // axes box + title. Return plot region.
    ctx.save();
    ctx.fillStyle="rgba(241,245,249,0.55)";
    ctx.fillRect(x,y,w,h);
    ctx.strokeStyle="rgba(15,23,42,0.12)";
    ctx.strokeRect(x,y,w,h);

    const mL=42, mR=10, mT=22, mB=22;
    const px=x+mL, py=y+mT, pw=w-mL-mR, ph=h-mT-mB;

    // title
    ctx.fillStyle="rgba(15,23,42,0.78)";
    ctx.font="800 12px ui-sans-serif, system-ui";
    ctx.fillText(opt.title||"", x+10, y+16);

    // ticks
    ctx.font="11px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillStyle="rgba(71,85,105,0.95)";

    const xt=opt.xTicks||{min:0,max:1,fmt:(v)=>v.toFixed(0)};
    const yt=opt.yTicks||{min:0,max:1,fmt:(v)=>v.toFixed(2)};
    const nx=4, ny=3;

    // x ticks
    for(let i=0;i<=nx;i++){
      const t=i/nx;
      const xv=xt.min + t*(xt.max-xt.min);
      const xp=px + t*pw;
      ctx.strokeStyle="rgba(15,23,42,0.06)";
      ctx.beginPath(); ctx.moveTo(xp,py); ctx.lineTo(xp,py+ph); ctx.stroke();
      const s=xt.fmt(xv);
      ctx.fillText(s, xp-ctx.measureText(s).width/2, y+h-6);
    }

    // y ticks
    for(let i=0;i<=ny;i++){
      const t=i/ny;
      const yv=yt.max - t*(yt.max-yt.min);
      const yp=py + t*ph;
      ctx.strokeStyle="rgba(15,23,42,0.06)";
      ctx.beginPath(); ctx.moveTo(px,yp); ctx.lineTo(px+pw,yp); ctx.stroke();
      const s=yt.fmt(yv);
      ctx.fillText(s, x+6, yp+4);
    }

    ctx.restore();
    return {plotX0:px, plotY0:py, plotW:pw, plotH:ph};
  }
  function plotLine(ctx, box, xs, ys, xmin,xmax,ymin,ymax, style){
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();
    ctx.strokeStyle=style.stroke||"#111";
    ctx.lineWidth=style.width||1.5;
    if(style.dash) ctx.setLineDash(style.dash); else ctx.setLineDash([]);
    let started=false;
    for(let i=0;i<xs.length;i++){
      const x=xs[i], y=ys[i];
      if(!Number.isFinite(x)||!Number.isFinite(y)) continue;
      const u=(x-xmin)/(xmax-xmin);
      const v=(y-ymin)/(ymax-ymin);
      const px=box.x+u*box.w;
      const py=box.y+box.h-(v*box.h);
      if(!started){ ctx.moveTo(px,py); started=true; }
      else ctx.lineTo(px,py);
    }
    if(started) ctx.stroke();
    ctx.restore();
  }
  function plotScatter(ctx, box, xs, ys, xmin,xmax,ymin,ymax, style, perPointColors){
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.clip();
    const r=style.r||2;
    for(let i=0;i<xs.length;i++){
      const x=xs[i], y=ys[i];
      if(!Number.isFinite(x)||!Number.isFinite(y)) continue;
      const u=(x-xmin)/(xmax-xmin);
      const v=(y-ymin)/(ymax-ymin);
      const px=box.x+u*box.w;
      const py=box.y+box.h-(v*box.h);
      ctx.beginPath();
      ctx.arc(px,py,r,0,2*Math.PI);
      ctx.fillStyle = perPointColors ? perPointColors[i] : (style.fill||"rgba(37,99,235,0.55)");
      ctx.fill();
    }
    ctx.restore();
  }

  function colorsFromStationIds(stIds, alpha=0.85){
    const out=new Array(stIds.length);
    for(let i=0;i<stIds.length;i++){
      const id=stIds[i];
      const c = (id>=1 && id<=12)? ST_COLORS[id-1] : "#2563eb";
      const r=parseInt(c.slice(1,3),16), g=parseInt(c.slice(3,5),16), b=parseInt(c.slice(5,7),16);
      out[i]=`rgba(${r},${g},${b},${alpha})`;
    }
    return out;
  }

  // -----------------------------
  // Orbit view camera + interactions
  // -----------------------------
  const cam = {
    kmPerPx: 20, // zoom
    cx: 0, cy: 0, // world center
    dragging:false,
    lastX:0, lastY:0,
  };

  function worldToScreen(X,Y,W,H){
    const x = (X - cam.cx)/cam.kmPerPx + W/2;
    const y = (-(Y - cam.cy))/cam.kmPerPx + H/2;
    return [x,y];
  }
  function screenToWorld(x,y,W,H){
    const X = (x - W/2)*cam.kmPerPx + cam.cx;
    const Y = (-(y - H/2))*cam.kmPerPx + cam.cy;
    return [X,Y];
  }

  function setupOrbitInteractions(){
    el.orbitCanvas.addEventListener("mousedown", (e)=>{
      cam.dragging=true; cam.lastX=e.clientX; cam.lastY=e.clientY;
    });
    window.addEventListener("mouseup", ()=>{cam.dragging=false;});
    window.addEventListener("mousemove", (e)=>{
      if(!cam.dragging) return;
      const dx=e.clientX-cam.lastX, dy=e.clientY-cam.lastY;
      cam.lastX=e.clientX; cam.lastY=e.clientY;
      cam.cx -= dx*cam.kmPerPx;
      cam.cy += dy*cam.kmPerPx;
      app.viewDirty=true;
    }, {passive:true});

    el.orbitCanvas.addEventListener("wheel", (e)=>{
      e.preventDefault();
      const rect=el.orbitCanvas.getBoundingClientRect();
      const mx=(e.clientX-rect.left), my=(e.clientY-rect.top);
      const ctx=setupHiDPICanvas(el.orbitCanvas);
      const W=rect.width, H=rect.height;
      const [wx,wy]=screenToWorld(mx,my,W,H);

      const factor = e.deltaY<0 ? 0.9 : 1.1;
      cam.kmPerPx = clamp(cam.kmPerPx*factor, 0.5, 500);

      // keep mouse point fixed
      const [sx,sy]=worldToScreen(wx,wy,W,H);
      cam.cx += (mx - sx)*cam.kmPerPx;
      cam.cy -= (my - sy)*cam.kmPerPx;

      app.viewDirty=true;
    }, {passive:false});
  }

  function resetView(){
    cam.kmPerPx=20;
    cam.cx=0; cam.cy=0;
    app.viewDirty=true;
  }
  function zoomToCraft(){
    cam.cx = app.truth[0];
    cam.cy = app.truth[2];
    cam.kmPerPx = 10;
    app.viewDirty=true;
  }

  // -----------------------------
  // App state
  // -----------------------------
  const app = {
    running:false,
    t:0,
    dt:5,
    speed:1,
    measEvery:1,
    stepCount:0,

    // states
    truth: [R_E+300, 0, 0, 0],
    ref:   [R_E+300, 0, 0, 0],
    uCmd:  [0,0],

    // EKF
    ekf: null,

    // modules
    dist: new Disturbance(),
    ctrl: new Controller(),

    // histories (fixed length)
    maxHist: 1400,
    tHist: [],
    xTrueHist: [[],[],[],[]],
    xHatHist:  [[],[],[],[]],
    xRefHist:  [[],[],[],[]],
    errHist:   [[],[],[],[]],
    sigHist:   [[],[],[],[]],
    neesHist: [],
    nisHist: [],
    nisDfHist: [],

    // measurements (scatter)
    measT: [], measRho: [], measRhod: [], measPhi: [], measStId: [],
    predT: [], predRho: [], predRhod: [], predPhi: [], predStId: [],

    // control
    uHistX: [], uHistY: [], uHistMagG: [], dvHist: [],
    aHatHist: [], eHatHist: [], aRefHist: [], eRefHist: [],

    // orbit view visuals
    pulses: [],
    squiggles: [],
    viewDirty:true,
    plotDirty:true,

    // desired orbit params (reference)
    tgtAlt: 300,
    tgtE: 0,
    tgtW: 0,

    dvTotal_ms: 0,

    // station tiles
    stationTiles: [],
  };

  function resetHist(){
    app.tHist.length=0;
    for(const a of app.xTrueHist) a.length=0;
    for(const a of app.xHatHist) a.length=0;
    for(const a of app.xRefHist) a.length=0;
    for(const a of app.errHist) a.length=0;
    for(const a of app.sigHist) a.length=0;
    app.neesHist.length=0;
    app.nisHist.length=0;
    app.nisDfHist.length=0;

    app.measT.length=app.measRho.length=app.measRhod.length=app.measPhi.length=app.measStId.length=0;
    app.predT.length=app.predRho.length=app.predRhod.length=app.predPhi.length=app.predStId.length=0;

    app.uHistX.length=app.uHistY.length=app.uHistMagG.length=app.dvHist.length=0;
    app.aHatHist.length=app.eHatHist.length=app.aRefHist.length=app.eRefHist.length=0;

    app.pulses.length=0;
    app.squiggles.length=0;
    app.dvTotal_ms=0;
  }

  function pushHist(){
    const t=app.t;
    app.tHist.push(t);
    for(let i=0;i<4;i++){
      app.xTrueHist[i].push(app.truth[i]);
      app.xHatHist[i].push(app.ekf.x[i]);
      app.xRefHist[i].push(app.ref[i]);
      const err = app.ekf.x[i] - app.truth[i];
      app.errHist[i].push(err);
      app.sigHist[i].push(Math.sqrt(Math.max(0, app.ekf.P[i][i])));
    }

    // NEES (truth known)
    const e = [app.truth[0]-app.ekf.x[0], app.truth[1]-app.ekf.x[1], app.truth[2]-app.ekf.x[2], app.truth[3]-app.ekf.x[3]];
    let nees = NaN;
    try{
      const Pinv = matInv(app.ekf.P);
      const tmp = matMulVec(Pinv, e);
      nees = e[0]*tmp[0]+e[1]*tmp[1]+e[2]*tmp[2]+e[3]*tmp[3];
    } catch(e){}
    app.neesHist.push(nees);

    // orbit elements for control plot
    const oeHat = oeFromState(app.ekf.x[0],app.ekf.x[1],app.ekf.x[2],app.ekf.x[3]);
    const oeRef = oeFromState(app.ref[0],app.ref[1],app.ref[2],app.ref[3]);
    app.aHatHist.push(oeHat.a);
    app.eHatHist.push(oeHat.e);
    app.aRefHist.push(oeRef.a);
    app.eRefHist.push(oeRef.e);

    // control histories
    const umag = hypot2(app.uCmd[0], app.uCmd[1]);
    const umag_g = (umag*1000)/G0;
    app.uHistX.push(app.uCmd[0]);
    app.uHistY.push(app.uCmd[1]);
    app.uHistMagG.push(umag_g);
    app.dvHist.push(app.dvTotal_ms);

    // trim
    const cap=app.maxHist;
    function trim(arr){ if(arr.length>cap) arr.splice(0, arr.length-cap); }
    trim(app.tHist);
    for(const a of app.xTrueHist) trim(a);
    for(const a of app.xHatHist) trim(a);
    for(const a of app.xRefHist) trim(a);
    for(const a of app.errHist) trim(a);
    for(const a of app.sigHist) trim(a);
    trim(app.neesHist); trim(app.nisHist); trim(app.nisDfHist);
    trim(app.uHistX); trim(app.uHistY); trim(app.uHistMagG); trim(app.dvHist);
    trim(app.aHatHist); trim(app.eHatHist); trim(app.aRefHist); trim(app.eRefHist);

    app.plotDirty=true;
  }

  // -----------------------------
  // Measurement buffers
  // -----------------------------
  function pushMeas(t, ids, y, yhat){
    for(let k=0;k<ids.length;k++){
      const id=ids[k];
      app.measT.push(t);
      app.measStId.push(id);
      app.measRho.push(y[3*k]);
      app.measRhod.push(y[3*k+1]);
      app.measPhi.push(y[3*k+2]);

      app.predT.push(t);
      app.predStId.push(id);
      app.predRho.push(yhat[3*k]);
      app.predRhod.push(yhat[3*k+1]);
      app.predPhi.push(yhat[3*k+2]);
    }
    const cap=app.maxHist*2;
    function trim(arr){ if(arr.length>cap) arr.splice(0, arr.length-cap); }
    trim(app.measT); trim(app.measStId); trim(app.measRho); trim(app.measRhod); trim(app.measPhi);
    trim(app.predT); trim(app.predStId); trim(app.predRho); trim(app.predRhod); trim(app.predPhi);
  }

  // -----------------------------
  // Rendering: orbit
  // -----------------------------
  function drawOrbit(){
    const ctx = setupHiDPICanvas(el.orbitCanvas);
    const rect = el.orbitCanvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;

    // background already via CSS; clear to transparent so gradient shows
    ctx.clearRect(0,0,W,H);

    // stars
    ctx.save();
    ctx.globalAlpha=0.35;
    for(let i=0;i<40;i++){
      const x = (i*97 % 1000)/1000 * W;
      const y = (i*223 % 1000)/1000 * H;
      ctx.fillStyle="rgba(255,255,255,0.35)";
      ctx.fillRect(x,y,1,1);
    }
    ctx.restore();

    // Earth
    const [ex,ey]=worldToScreen(0,0,W,H);
    const RePx = R_E / cam.kmPerPx;
    const grad = ctx.createRadialGradient(ex,ey,RePx*0.2, ex,ey,RePx*1.05);
    grad.addColorStop(0,"rgba(37,99,235,0.38)");
    grad.addColorStop(1,"rgba(11,18,32,0.95)");
    ctx.beginPath();
    ctx.arc(ex,ey,RePx,0,2*Math.PI);
    ctx.fillStyle=grad;
    ctx.fill();
    ctx.strokeStyle="rgba(249,115,22,0.30)";
    ctx.lineWidth=1.2;
    ctx.stroke();

    // reference orbit path (dotted)
    ctx.save();
    ctx.setLineDash([6,6]);
    ctx.strokeStyle="rgba(6,182,212,0.75)";
    ctx.lineWidth=1.2;
    ctx.beginPath();
    for(let i=0;i<app.xRefHist[0].length;i++){
      const X=app.xRefHist[0][i], Y=app.xRefHist[2][i];
      const [sx,sy]=worldToScreen(X,Y,W,H);
      if(i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
    }
    ctx.stroke();
    ctx.restore();

    // truth path
    ctx.save();
    ctx.strokeStyle="rgba(255,255,255,0.65)";
    ctx.lineWidth=1.6;
    ctx.beginPath();
    for(let i=0;i<app.xTrueHist[0].length;i++){
      const X=app.xTrueHist[0][i], Y=app.xTrueHist[2][i];
      const [sx,sy]=worldToScreen(X,Y,W,H);
      if(i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
    }
    ctx.stroke();
    ctx.restore();

    // estimate path
    ctx.save();
    ctx.setLineDash([6,4]);
    ctx.strokeStyle="rgba(37,99,235,0.90)";
    ctx.lineWidth=1.6;
    ctx.beginPath();
    let started=false;
    for(let i=0;i<app.xHatHist[0].length;i++){
      const X=app.xHatHist[0][i], Y=app.xHatHist[2][i];
      if(!Number.isFinite(X)||!Number.isFinite(Y)) continue;
      const [sx,sy]=worldToScreen(X,Y,W,H);
      if(!started){ ctx.moveTo(sx,sy); started=true; }
      else ctx.lineTo(sx,sy);
    }
    if(started) ctx.stroke();
    ctx.restore();

    // stations + links for visible stations
    const vis = visibleStations(app.truth, app.t);
    for(let i=0;i<12;i++){
      const s=stationState(i, app.t);
      const [sx,sy]=worldToScreen(s.X,s.Y,W,H);

      ctx.beginPath();
      ctx.arc(sx,sy,4.2,0,2*Math.PI);
      ctx.fillStyle=ST_COLORS[i];
      ctx.globalAlpha=0.9;
      ctx.fill();
      ctx.globalAlpha=1;
    }

    // pulses & squiggles for measuring stations
    drawSignals(ctx,W,H);

    // markers: truth, EKF, ref
    drawMarker(ctx, app.truth[0], app.truth[2], W,H, 5.0, "rgba(249,115,22,1)");
    drawMarker(ctx, app.ekf.x[0], app.ekf.x[2], W,H, 5.0, "rgba(37,99,235,1)");
    drawMarker(ctx, app.ref[0], app.ref[2], W,H, 4.5, "rgba(6,182,212,0.95)");

    // covariance ellipse (position)
    drawCovEllipse(ctx,W,H);

    // small overlay text
    ctx.save();
    ctx.fillStyle="rgba(255,255,255,0.85)";
    ctx.font="800 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`km/px: ${fmt(cam.kmPerPx,1)}`, 12, 18);
    ctx.restore();
  }

  function drawMarker(ctx,X,Y,W,H,r,fill){
    const [sx,sy]=worldToScreen(X,Y,W,H);
    ctx.beginPath();
    ctx.arc(sx,sy,r,0,2*Math.PI);
    ctx.fillStyle=fill;
    ctx.fill();
    ctx.strokeStyle="rgba(15,23,42,0.35)";
    ctx.lineWidth=1.1;
    ctx.stroke();
  }

  function drawCovEllipse(ctx,W,H){
    const P=app.ekf.P;
    const Pxx=P[0][0], Pxy=P[0][2], Pyy=P[2][2];
    if(!Number.isFinite(Pxx)||!Number.isFinite(Pyy)) return;
    // eigen of 2x2
    const tr=Pxx+Pyy;
    const det=Pxx*Pyy-Pxy*Pxy;
    const disc=Math.max(0,tr*tr/4-det);
    const l1=tr/2 + Math.sqrt(disc);
    const l2=tr/2 - Math.sqrt(disc);
    if(l1<=0||l2<=0) return;
    const angle=0.5*Math.atan2(2*Pxy,(Pxx-Pyy));
    const a=2*Math.sqrt(l1); // 2-sigma
    const b=2*Math.sqrt(l2);

    const [sx,sy]=worldToScreen(app.ekf.x[0], app.ekf.x[2], W,H);
    ctx.save();
    ctx.translate(sx,sy);
    ctx.rotate(-angle); // screen y down handled in worldToScreen
    ctx.beginPath();
    ctx.ellipse(0,0, a/cam.kmPerPx, b/cam.kmPerPx, 0, 0, 2*Math.PI);
    ctx.strokeStyle="rgba(234,88,12,0.65)";
    ctx.lineWidth=1.2;
    ctx.setLineDash([5,4]);
    ctx.stroke();
    ctx.restore();
  }

  function drawSignals(ctx,W,H){
    // pulses: fade circles on stations
    const pulsesKeep=[];
    for(const p of app.pulses){
      p.age += app.dt;
      if(p.age > p.life) continue;
      pulsesKeep.push(p);

      const s=stationState(p.sid-1, app.t);
      const [sx,sy]=worldToScreen(s.X,s.Y,W,H);
      const r = 6 + 18*(p.age/p.life);
      const a = 1 - (p.age/p.life);
      ctx.beginPath();
      ctx.arc(sx,sy,r,0,2*Math.PI);
      ctx.strokeStyle = `rgba(${p.rgb.r},${p.rgb.g},${p.rgb.b},${0.35*a})`;
      ctx.lineWidth=1.3;
      ctx.stroke();
    }
    app.pulses = pulsesKeep;

    // squiggles
    const squKeep=[];
    for(const s of app.squiggles){
      s.age += app.dt;
      if(s.age > s.life) continue;
      squKeep.push(s);

      const p0=worldToScreen(s.x0,s.y0,W,H);
      const p1=worldToScreen(s.x1,s.y1,W,H);
      const dx=p1[0]-p0[0], dy=p1[1]-p0[1];
      const L=Math.hypot(dx,dy);
      if(L<1) continue;
      const ux=dx/L, uy=dy/L;
      const nx=-uy, ny=ux;

      const a = 1 - (s.age/s.life);
      ctx.strokeStyle = `rgba(${s.rgb.r},${s.rgb.g},${s.rgb.b},${0.55*a})`;
      ctx.lineWidth=1.5;
      ctx.beginPath();
      const amp=4;
      const waves=8;
      for(let i=0;i<=waves;i++){
        const t=i/waves;
        const phase = 2*Math.PI*(t*2 + s.phase);
        const ox = p0[0] + t*dx + amp*Math.sin(phase)*nx;
        const oy = p0[1] + t*dy + amp*Math.sin(phase)*ny;
        if(i===0) ctx.moveTo(ox,oy); else ctx.lineTo(ox,oy);
      }
      ctx.stroke();
    }
    app.squiggles = squKeep;
  }

  // -----------------------------
  // Plots
  // -----------------------------
  function drawPlots(){
    drawStatesPlot();
    drawErrorsPlot();
    drawMeasPlot();
    drawTrajPlot();
    drawConsPlot();
    drawCtrlPlot();
    app.plotDirty=false;
  }

  function drawStatesPlot(){
    const ctx=setupHiDPICanvas(el.plotStates);
    const rect=el.plotStates.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);
    if(app.tHist.length<2) return;

    const pad=12;
    const tileW=(W-pad*3)/2;
    const tileH=(H-pad*3)/2;
    const labels=["X (km)","Ẋ (km/s)","Y (km)","Ẏ (km/s)"];
    const t=app.tHist; const tmin=t[0], tmax=t[t.length-1];

    for(let i=0;i<4;i++){
      const col=i%2,row=Math.floor(i/2);
      const x=pad+col*(tileW+pad);
      const y=pad+row*(tileH+pad);

      const truth=app.xTrueHist[i];
      const est=app.xHatHist[i];
      const ref=app.xRefHist[i];
      const sig=app.sigHist[i];
      const up=sig.map(s=> (Number.isFinite(s)? (est[sig.indexOf(s)] + 2*s): NaN)); // not used
      const lo=sig.map(s=> (Number.isFinite(s)? (est[sig.indexOf(s)] - 2*s): NaN)); // not used

      // build 2sigma bands efficiently
      const up2=new Array(est.length), lo2=new Array(est.length);
      for(let k=0;k<est.length;k++){ up2[k]=est[k]+2*sig[k]; lo2[k]=est[k]-2*sig[k]; }

      const {mn,mx}=computeMinMax([truth,est,ref,up2,lo2]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{
        title: labels[i],
        xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)},
        yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v, (i%2===0)?0:3)}
      });
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,truth,tmin,tmax,mn,mx,{stroke:"rgba(15,23,42,0.65)",width:1.5});
      plotLine(ctx,box,t,ref,  tmin,tmax,mn,mx,{stroke:"rgba(6,182,212,0.80)",width:1.2,dash:[5,5]});
      plotLine(ctx,box,t,est,  tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.90)",width:1.5});
      plotLine(ctx,box,t,up2,  tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
      plotLine(ctx,box,t,lo2,  tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
    }
  }

  function drawErrorsPlot(){
    const ctx=setupHiDPICanvas(el.plotErrors);
    const rect=el.plotErrors.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);
    if(app.tHist.length<2) return;

    const pad=12;
    const tileW=W-pad*2;
    const tileH=(H-pad*5)/4;
    const labels=["X error (km)","Ẋ error (km/s)","Y error (km)","Ẏ error (km/s)"];
    const t=app.tHist; const tmin=t[0], tmax=t[t.length-1];

    for(let i=0;i<4;i++){
      const x=pad, y=pad+i*(tileH+pad);
      const err=app.errHist[i];
      const sig=app.sigHist[i];
      const up=new Array(err.length), lo=new Array(err.length);
      for(let k=0;k<err.length;k++){ up[k]= 2*sig[k]; lo[k]= -2*sig[k]; }
      const {mn,mx}=computeMinMax([err,up,lo]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{
        title: labels[i],
        xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)},
        yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,3)}
      });
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,err,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.90)",width:1.5});
      plotLine(ctx,box,t,up, tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
      plotLine(ctx,box,t,lo, tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
    }
  }

  function drawMeasPlot(){
    const ctx=setupHiDPICanvas(el.plotMeas);
    const rect=el.plotMeas.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);

    const pad=12;
    const tileW=W-pad*2;
    const tileH=(H-pad*5)/3;

    const tmin = Math.min(app.tHist[0]??0, app.measT[0]??0, app.predT[0]??0);
    const tmax = Math.max(app.tHist.at(-1)??1, app.measT.at(-1)??1, app.predT.at(-1)??1);
    if(!(tmax>tmin)) return;

    // range
    {
      const x=pad,y=pad;
      const {mn,mx}=computeMinMax([app.measRho, app.predRho]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Range ρ (km) — measured vs predicted", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,0)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      const c=colorsFromStationIds(app.measStId,0.9);
      plotScatter(ctx,box,app.measT,app.measRho,tmin,tmax,mn,mx,{r:2.3},c);
      plotScatter(ctx,box,app.predT,app.predRho,tmin,tmax,mn,mx,{r:2.0,fill:"rgba(15,23,42,0.25)"});
    }
    // rhod
    {
      const x=pad,y=pad+tileH+pad;
      const {mn,mx}=computeMinMax([app.measRhod, app.predRhod]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Range-rate ρ̇ (km/s) — measured vs predicted", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,4)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      const c=colorsFromStationIds(app.measStId,0.9);
      plotScatter(ctx,box,app.measT,app.measRhod,tmin,tmax,mn,mx,{r:2.3},c);
      plotScatter(ctx,box,app.predT,app.predRhod,tmin,tmax,mn,mx,{r:2.0,fill:"rgba(15,23,42,0.25)"});
    }
    // phi
    {
      const x=pad,y=pad+2*(tileH+pad);
      const {mn,mx}=computeMinMax([app.measPhi, app.predPhi]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Angle φ (rad) — measured vs predicted", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,3)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      const c=colorsFromStationIds(app.measStId,0.9);
      plotScatter(ctx,box,app.measT,app.measPhi,tmin,tmax,mn,mx,{r:2.3},c);
      plotScatter(ctx,box,app.predT,app.predPhi,tmin,tmax,mn,mx,{r:2.0,fill:"rgba(15,23,42,0.25)"});
    }
  }

  function drawTrajPlot(){
    const ctx=setupHiDPICanvas(el.plotTraj);
    const rect=el.plotTraj.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);
    if(app.tHist.length<2) return;

    const pad=12;
    const x0=pad,y0=pad,w=W-pad*2,h=H-pad*2;

    // extents from truth+est+ref
    const Xs=[...app.xTrueHist[0], ...app.xHatHist[0], ...app.xRefHist[0]].filter(Number.isFinite);
    const Ys=[...app.xTrueHist[2], ...app.xHatHist[2], ...app.xRefHist[2]].filter(Number.isFinite);
    const mmx=computeMinMax([Xs]);
    const mmy=computeMinMax([Ys]);

    let xmin=mmx.mn, xmax=mmx.mx, ymin=mmy.mn, ymax=mmy.mx;
    const xr=xmax-xmin, yr=ymax-ymin;
    const r=Math.max(xr,yr);
    const xc=0.5*(xmin+xmax), yc=0.5*(ymin+ymax);
    xmin=xc-0.55*r; xmax=xc+0.55*r; ymin=yc-0.55*r; ymax=yc+0.55*r;

    const ax=drawAxes(ctx,x0,y0,w,h,{title:"Inertial XY trajectory (km) — truth/est/ref + stations", xTicks:{min:xmin,max:xmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:ymin,max:ymax,fmt:(v)=>fmt(v,0)}});
    const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
    const sx=(X)=>box.x+(X-xmin)/(xmax-xmin)*box.w;
    const sy=(Y)=>box.y+box.h-(Y-ymin)/(ymax-ymin)*box.h;

    ctx.save();
    ctx.beginPath(); ctx.rect(box.x,box.y,box.w,box.h); ctx.clip();

    // earth
    const RePx = (R_E/(xmax-xmin))*box.w;
    ctx.beginPath(); ctx.arc(sx(0),sy(0),RePx,0,2*Math.PI);
    ctx.fillStyle="rgba(249,115,22,0.08)"; ctx.fill();
    ctx.strokeStyle="rgba(249,115,22,0.22)"; ctx.lineWidth=1.2; ctx.stroke();

    // stations
    for(let i=0;i<12;i++){
      const s=stationState(i, app.t);
      ctx.beginPath();
      ctx.arc(sx(s.X),sy(s.Y),2.6,0,2*Math.PI);
      ctx.fillStyle=ST_COLORS[i]; ctx.globalAlpha=0.9; ctx.fill(); ctx.globalAlpha=1;
    }

    // paths
    const t=app.tHist;
    ctx.strokeStyle="rgba(15,23,42,0.65)"; ctx.lineWidth=1.6;
    ctx.beginPath();
    for(let i=0;i<t.length;i++){ ctx.lineTo(sx(app.xTrueHist[0][i]), sy(app.xTrueHist[2][i])); }
    ctx.stroke();

    ctx.save();
    ctx.setLineDash([6,4]);
    ctx.strokeStyle="rgba(37,99,235,0.85)"; ctx.lineWidth=1.5;
    ctx.beginPath();
    for(let i=0;i<t.length;i++){
      const X=app.xHatHist[0][i], Y=app.xHatHist[2][i];
      if(i===0) ctx.moveTo(sx(X),sy(Y)); else ctx.lineTo(sx(X),sy(Y));
    }
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.setLineDash([6,6]);
    ctx.strokeStyle="rgba(6,182,212,0.80)"; ctx.lineWidth=1.3;
    ctx.beginPath();
    for(let i=0;i<t.length;i++){
      ctx.lineTo(sx(app.xRefHist[0][i]), sy(app.xRefHist[2][i]));
    }
    ctx.stroke();
    ctx.restore();

    ctx.restore();
  }

  function drawConsPlot(){
    const ctx=setupHiDPICanvas(el.plotCons);
    const rect=el.plotCons.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);
    if(app.tHist.length<2) return;

    const pad=12;
    const tileW=W-pad*2;
    const tileH=(H-pad*4)/2;
    const alpha = clamp(parseFloat(el.inpAlpha.value)||0.05, 0.01, 0.2);
    const t=app.tHist; const tmin=t[0], tmax=t[t.length-1];

    // NEES df=4
    {
      const x=pad,y=pad;
      const lo=chi2Inv(alpha/2,4), hi=chi2Inv(1-alpha/2,4);
      const ylo=new Array(t.length).fill(lo), yhi=new Array(t.length).fill(hi);
      const {mn,mx}=computeMinMax([app.neesHist,ylo,yhi]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"NEES (df=4) with χ² bounds", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,1)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,app.neesHist,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.90)",width:1.5});
      plotLine(ctx,box,t,ylo,tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
      plotLine(ctx,box,t,yhi,tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
    }

    // NIS df varies
    {
      const x=pad,y=pad+tileH+pad;
      const ylo=[], yhi=[];
      for(let i=0;i<app.nisDfHist.length;i++){
        const df=app.nisDfHist[i];
        if(!(df>0)){ ylo.push(NaN); yhi.push(NaN); continue; }
        ylo.push(chi2Inv(alpha/2,df));
        yhi.push(chi2Inv(1-alpha/2,df));
      }
      const {mn,mx}=computeMinMax([app.nisHist,ylo,yhi]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"NIS with χ² bounds (df=3×visible)", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,1)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,app.nisHist,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.90)",width:1.5});
      plotLine(ctx,box,t,ylo,tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
      plotLine(ctx,box,t,yhi,tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.70)",width:1.1,dash:[6,4]});
    }
  }

  function drawCtrlPlot(){
    const ctx=setupHiDPICanvas(el.plotCtrl);
    const rect=el.plotCtrl.getBoundingClientRect();
    const W=rect.width,H=rect.height;
    clearPanel(ctx,W,H);
    if(app.tHist.length<2) return;

    const pad=12;
    const tileW=(W-pad*3)/2;
    const tileH=(H-pad*3)/2;

    const t=app.tHist; const tmin=t[0], tmax=t[t.length-1];
    const umaxG = parseFloat(el.inpUMaxG.value)||0.01;

    // u components
    {
      const x=pad,y=pad;
      const uX=app.uHistX, uY=app.uHistY;
      const {mn,mx}=computeMinMax([uX,uY]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Control accel components (km/s²)", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,6)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,uX,tmin,tmax,mn,mx,{stroke:"rgba(249,115,22,0.9)",width:1.5});
      plotLine(ctx,box,t,uY,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.9)",width:1.5});
    }

    // u magnitude + dv
    {
      const x=pad+tileW+pad,y=pad;
      const um=app.uHistMagG;
      const bound=new Array(um.length).fill(umaxG);
      const {mn,mx}=computeMinMax([um,bound]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"|u| (g) + cumulative Δv (m/s)", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:Math.min(mn,-0.001),max:mx,fmt:(v)=>fmt(v,3)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,um,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.9)",width:1.5});
      plotLine(ctx,box,t,bound,tmin,tmax,mn,mx,{stroke:"rgba(234,88,12,0.7)",width:1.1,dash:[6,4]});
      // Δv scaled to overlay? (simple: secondary trace normalized into same axis)
      const dv=app.dvHist;
      if(dv.length){
        const dvMax=Math.max(...dv,1);
        const dvScaled=dv.map(v=> (v/dvMax)*mx*0.95);
        plotLine(ctx,box,t,dvScaled,tmin,tmax,mn,mx,{stroke:"rgba(6,182,212,0.85)",width:1.2,dash:[4,4]});
      }
    }

    // a tracking
    {
      const x=pad,y=pad+tileH+pad;
      const aH=app.aHatHist, aR=app.aRefHist;
      const {mn,mx}=computeMinMax([aH,aR]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Semi-major axis a (km) — EKF vs ref", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,0)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,aR,tmin,tmax,mn,mx,{stroke:"rgba(6,182,212,0.85)",width:1.5});
      plotLine(ctx,box,t,aH,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.85)",width:1.5});
    }

    // e tracking
    {
      const x=pad+tileW+pad,y=pad+tileH+pad;
      const eH=app.eHatHist, eR=app.eRefHist;
      const {mn,mx}=computeMinMax([eH,eR]);
      const ax=drawAxes(ctx,x,y,tileW,tileH,{title:"Eccentricity e — EKF vs ref", xTicks:{min:tmin,max:tmax,fmt:(v)=>fmt(v,0)}, yTicks:{min:mn,max:mx,fmt:(v)=>fmt(v,3)}});
      const box={x:ax.plotX0,y:ax.plotY0,w:ax.plotW,h:ax.plotH};
      plotLine(ctx,box,t,eR,tmin,tmax,mn,mx,{stroke:"rgba(6,182,212,0.85)",width:1.5});
      plotLine(ctx,box,t,eH,tmin,tmax,mn,mx,{stroke:"rgba(37,99,235,0.85)",width:1.5});
    }
  }

  // -----------------------------
  // Tabs
  // -----------------------------
  function setupTabs(){
    const tabs=document.querySelectorAll(".tab");
    const panes=document.querySelectorAll(".pane");
    tabs.forEach(tab=>{
      tab.addEventListener("click", ()=>{
        const name=tab.getAttribute("data-tab");
        tabs.forEach(t=>t.classList.remove("is-active"));
        tab.classList.add("is-active");
        panes.forEach(p=>{
          p.classList.toggle("is-active", p.getAttribute("data-pane")===name);
        });
        app.plotDirty=true;
      });
    });
  }

  // Plot pan/zoom (pause-only)
  const plotView = {scale:1, ox:0, oy:0};
  function setupPlotPanZoom(){
    const canvases=[el.plotStates,el.plotErrors,el.plotMeas,el.plotTraj,el.plotCons,el.plotCtrl];
    canvases.forEach(cv=>{
      let dragging=false, lx=0, ly=0;
      cv.addEventListener("mousedown",(e)=>{
        if(app.running) return;
        dragging=true; lx=e.offsetX; ly=e.offsetY;
      });
      cv.addEventListener("mousemove",(e)=>{
        if(!dragging) return;
        plotView.ox += (e.offsetX-lx);
        plotView.oy += (e.offsetY-ly);
        lx=e.offsetX; ly=e.offsetY;
        app.plotDirty=true;
      });
      window.addEventListener("mouseup",()=>dragging=false);
      cv.addEventListener("wheel",(e)=>{
        if(app.running) return;
        e.preventDefault();
        const f=e.deltaY<0?1.08:0.92;
        plotView.scale = clamp(plotView.scale*f, 0.5, 3);
        app.plotDirty=true;
      }, {passive:false});
    });
  }
  // NOTE: For simplicity, plotView isn't yet applied in mapping; this keeps code small and avoids clipping bugs.
  // If you want full plot pan/zoom, we can apply plotView in axes transforms later.

  // -----------------------------
  // UI
  // -----------------------------
  function updateTargetReadouts(params){
    el.readoutA.textContent = `${fmt(params.a,0)} km`;
    el.readoutRa.textContent = `${fmt(params.ra,0)} km`;
    el.readoutPeriod.textContent = `${fmt(params.T,0)} s`;
  }

  function syncFromUI(){
    app.dt = clamp(parseFloat(el.inpDt.value)||5, 0.1, 60);
    app.measEvery = clamp(parseInt(el.inpMeasEvery.value||"1",10)||1, 1, 60);
    el.readoutDt.textContent = `${fmt(app.dt,1)} s`;

    app.tgtAlt = clamp(parseFloat(el.inpTgtAlt.value)||300, 150, 40000);
    app.tgtE = clamp(parseFloat(el.inpTgtE.value)||0, 0, 0.6);
    app.tgtW = clamp(parseFloat(el.inpTgtW.value)||0, -180, 180);

    // update derived orbit readouts
    const thetaNow = Math.atan2(app.truth[2], app.truth[0]);
    const params = stateFromPerigeeAltE(app.tgtAlt, app.tgtE, app.tgtW, el.selPhase.value, thetaNow);
    updateTargetReadouts(params);
  }

  function applyTargetOrbit(){
    syncFromUI();
    const thetaNow = Math.atan2(app.truth[2], app.truth[0]);
    const params = stateFromPerigeeAltE(app.tgtAlt, app.tgtE, app.tgtW, el.selPhase.value, thetaNow);
    app.ref = [params.X, params.VX, params.Y, params.VY];
    app.ctrl.reset();
    app.plotDirty=true;
    app.viewDirty=true;
  }

  function holdHere(){
    // set ref to current truth orbit (same state)
    app.ref = app.truth.slice();
    // also update UI to match (approx perigee alt for current osculating)
    const oe = oeFromState(app.truth[0],app.truth[1],app.truth[2],app.truth[3]);
    const a=oe.a, e=oe.e;
    const rp = a*(1-e);
    el.inpTgtAlt.value = (rp - R_E).toFixed(0);
    el.inpTgtE.value = e.toFixed(2);
    el.inpTgtW.value = "0";
    app.ctrl.reset();
    syncFromUI();
  }

  function toggleRun(){
    app.running=!app.running;
    el.readoutRun.textContent = app.running? "RUNNING":"PAUSED";
    el.btnToggleRun.textContent = app.running? "Pause":"Run";
    if(app.running){ plotView.scale=1; plotView.ox=0; plotView.oy=0; }
  }

  function buildStationBoard(){
    el.stationBoard.innerHTML="";
    app.stationTiles.length=0;
    for(let i=0;i<12;i++){
      const tile=document.createElement("div");
      tile.className="stationTile";
      tile.style.borderLeftColor = ST_COLORS[i];
      tile.innerHTML = `
        <div class="stationTile__dish" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <path d="M12 3a9 9 0 0 1 9 9h-2a7 7 0 0 0-7-7V3z"></path>
            <path d="M12 7a5 5 0 0 1 5 5h-2a3 3 0 0 0-3-3V7z"></path>
            <path d="M13 12h-2v8h2v-8z"></path>
          </svg>
        </div>
        <div>
          <div class="stationTile__name">GS-${i+1}</div>
          <div class="stationTile__sub" id="stSub${i+1}">idle</div>
        </div>
      `;
      el.stationBoard.appendChild(tile);
      app.stationTiles.push(tile);
    }
  }

  function updateStationTiles(visIds){
    // reset all
    for(let i=0;i<12;i++){
      const sub=document.getElementById(`stSub${i+1}`);
      if(sub) sub.textContent="idle";
      app.stationTiles[i].style.opacity = "0.92";
    }
    for(const id of visIds){
      const sub=document.getElementById(`stSub${id}`);
      if(sub) sub.textContent="measuring";
      app.stationTiles[id-1].style.opacity = "1";
      // point dish toward craft
      const tile=app.stationTiles[id-1];
      const svg=tile.querySelector("svg");
      const s=stationState(id-1, app.t);
      const ang=Math.atan2(app.truth[2]-s.Y, app.truth[0]-s.X);
      svg.style.transform = `rotate(${ang}rad)`;
      svg.style.color = ST_COLORS[id-1];
    }
  }

  // -----------------------------
  // Simulation step
  // -----------------------------
  function makeMeasurements(ids){
    const m=ids.length*3;
    const y=new Float64Array(m);
    const yhat=new Float64Array(m);
    const Rdiag=new Float64Array(m);

    const sigRho = parseFloat(el.inpSigRho.value)||0.2;
    const sigRhod= parseFloat(el.inpSigRhod.value)||0.002;
    const sigPhi = parseFloat(el.inpSigPhi.value)||0.002;
    const noiseOn = !!el.tglMeasNoise.checked;

    for(let k=0;k<ids.length;k++){
      const s=stationState(ids[k]-1, app.t);
      const hT=hMeas(app.truth,s);
      const hE=hMeas(app.ekf.x,s);

      const n0 = noiseOn ? randn()*sigRho : 0;
      const n1 = noiseOn ? randn()*sigRhod: 0;
      const n2 = noiseOn ? randn()*sigPhi : 0;

      y[3*k]   = hT[0] + n0;
      y[3*k+1] = hT[1] + n1;
      y[3*k+2] = hT[2] + n2;

      yhat[3*k]   = hE[0];
      yhat[3*k+1] = hE[1];
      yhat[3*k+2] = hE[2];

      Rdiag[3*k]   = sigRho*sigRho;
      Rdiag[3*k+1] = sigRhod*sigRhod;
      Rdiag[3*k+2] = sigPhi*sigPhi;
    }
    return {y,yhat,Rdiag};
  }

  function simStep(){
    syncFromUI();
    const dt=app.dt;
    const measEvery=app.measEvery;

    // controller gains
    const agg = clamp(parseFloat(el.inpAgg.value)||1, 0.2, 3);
    const gains={
      Kp: (parseFloat(el.inpKp.value)||2e-6)*agg,
      Kd: (parseFloat(el.inpKd.value)||8e-4)*agg,
      Ki: (parseFloat(el.inpKi.value)||2e-9)*agg,
      iClamp: clamp(parseFloat(el.inpIClamp.value)||5000, 100, 5e5),
      lqrAgg: agg
    };
    const uMax_kmps2 = (clamp(parseFloat(el.inpUMaxG.value)||0.01, 0.0001, 0.05)*G0)/1000;

    // choose estimate for control
    const xForCtrl = (el.tglCtrlUseEKF.checked && app.ekf) ? app.ekf.x : app.truth;
    const uCtrl = app.ctrl.compute(dt, el.selCtrlMode.value, el.tglCtrl.checked, true, xForCtrl, app.ref, uMax_kmps2, gains);

    el.calloutCtrlSat.hidden = !app.ctrl.saturated;
    app.uCmd = uCtrl;

    // truth disturbance
    const distOn = !!el.tglTruthDist.checked;
    const distModel = el.selDistModel.value;
    const distSigma = parseFloat(el.inpDistSigma.value)||0.0002; // m/s^2
    const distTau = parseFloat(el.inpDistTau.value)||600;
    const d = app.dist.step(dt, distOn, distModel, distSigma, distTau);

    // show warning when enabled
    el.pillDist.hidden = !distOn;
    el.overlayWarn.hidden = !distOn;

    // propagate truth with uCtrl + disturbance
    const uTot = [uCtrl[0]+d[0], uCtrl[1]+d[1]];
    app.truth = rk4(app.truth, uTot, dt);

    // propagate reference (unforced)
    app.ref = rk4(app.ref, [0,0], dt);

    // EKF predict
    const qSigma = parseFloat(el.inpQsigma.value)||0.00005; // m/s^2
    app.ekf.predict(dt, uCtrl, qSigma);

    // measurements & update (optionally downsample)
    let nis = NaN, df=0;
    if(app.stepCount % measEvery === 0){
      const ids = visibleStations(app.truth, app.t);
      updateStationTiles(ids);

      // visuals for station links
      if(ids.length){
        for(const id of ids){
          const s=stationState(id-1, app.t);
          const c=ST_COLORS[id-1];
          const rgb={r:parseInt(c.slice(1,3),16), g:parseInt(c.slice(3,5),16), b:parseInt(c.slice(5,7),16)};
          app.pulses.push({sid:id, rgb, age:0, life:2.0});
          app.squiggles.push({x0:app.truth[0], y0:app.truth[2], x1:s.X, y1:s.Y, rgb, age:0, life:1.2, phase:Math.random()});
        }
      }

      const meas = makeMeasurements(ids);
      const out = app.ekf.update(meas.y, ids, meas.Rdiag);
      nis = out.nis;
      df = app.ekf.lastDf;

      // record for measurement plots
      pushMeas(app.t, ids, meas.y, meas.yhat);

      app.nisHist.push(nis);
      app.nisDfHist.push(df);
      // trim
      if(app.nisHist.length>app.maxHist) app.nisHist.splice(0, app.nisHist.length-app.maxHist);
      if(app.nisDfHist.length>app.maxHist) app.nisDfHist.splice(0, app.nisDfHist.length-app.maxHist);
      el.readoutVisible.textContent = String(ids.length);
    } else {
      // keep readout stable
      const ids = visibleStations(app.truth, app.t);
      el.readoutVisible.textContent = String(ids.length);
    }

    // crash guard
    const r=hypot2(app.truth[0],app.truth[2]);
    if(r < R_E){
      // auto freeze control and reset
      resetSim();
      return;
    }

    // Δv accumulation (m/s)
    const uMag = hypot2(uCtrl[0],uCtrl[1]);
    app.dvTotal_ms += (uMag*1000)*dt;
    el.readoutDV.textContent = `${fmt(app.dvTotal_ms,1)} m/s`;

    app.t += dt;
    app.stepCount++;

    // readouts
    el.readoutTime.textContent = `${fmt(app.t,1)} s`;

    // orbit readout: a,e of EKF
    const oe = oeFromState(app.ekf.x[0],app.ekf.x[1],app.ekf.x[2],app.ekf.x[3]);
    el.readoutOrbit.textContent = `EKF: a=${fmt(oe.a,0)} km, e=${fmt(oe.e,3)}`;

    pushHist();
  }

  function resetSim(){
    app.running=false;
    el.readoutRun.textContent="PAUSED";
    el.btnToggleRun.textContent="Run";

    app.t=0;
    app.stepCount=0;
    syncFromUI();

    // initial truth: start at perigee of target orbit (nice demo)
    const thetaNow = 0;
    const init = stateFromPerigeeAltE(app.tgtAlt, app.tgtE, app.tgtW, "perigee", thetaNow);
    app.truth = [init.X, init.VX, init.Y, init.VY];
    app.ref   = [init.X, init.VX, init.Y, init.VY];

    // EKF init with small offset + moderate covariance
    const x0 = [app.truth[0]+1.5, app.truth[1]+0.002, app.truth[2]-1.0, app.truth[3]-0.002];
    const P0 = [
      [25,0,0,0],
      [0,1e-4,0,0],
      [0,0,25,0],
      [0,0,0,1e-4],
    ];
    app.ekf = new EKF(x0, P0);

    app.dist = new Disturbance();
    app.ctrl.reset();
    resetHist();
    pushHist();

    resetView();
    app.plotDirty=true;
    app.viewDirty=true;
    el.readoutDV.textContent = "0.0 m/s";
  }

  // -----------------------------
  // Main loop
  // -----------------------------
  let lastFrame=performance.now();
  function loop(now){
    const dtms=now-lastFrame;
    lastFrame=now;

    if(app.running){
      for(let i=0;i<app.speed;i++) simStep();
      app.viewDirty=true;
    }

    // orbit redraw every frame (smooth signals)
    drawOrbit();

    // plots throttled
    if(app.plotDirty) drawPlots();

    requestAnimationFrame(loop);
  }

  // -----------------------------
  // Wire UI
  // -----------------------------
  function bindUI(){
    el.btnToggleRun.addEventListener("click", toggleRun);
    el.btnStep.addEventListener("click", ()=>{ simStep(); app.viewDirty=true; drawPlots(); });
    el.btnReset.addEventListener("click", resetSim);
    el.btnZoomCraft.addEventListener("click", zoomToCraft);
    el.btnResetView.addEventListener("click", resetView);
    el.btnApplyTarget.addEventListener("click", applyTargetOrbit);
    el.btnHoldHere.addEventListener("click", holdHere);

    document.querySelectorAll("[data-speed]").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        app.speed = parseInt(btn.getAttribute("data-speed"),10)||1;
        document.querySelectorAll("[data-speed]").forEach(b=>b.classList.remove("btn--primary"));
        btn.classList.add("btn--primary");
      });
    });

    // keep UI-derived readouts fresh
    ["inpDt","inpMeasEvery","inpTgtAlt","inpTgtE","inpTgtW","selPhase"].forEach(id=>{
      const node=document.getElementById(id);
      node && node.addEventListener("change", ()=>{ syncFromUI(); app.plotDirty=true; });
    });

    el.tglTruthDist.addEventListener("change", ()=>{ syncFromUI(); });
    el.selDistModel.addEventListener("change", ()=>{ syncFromUI(); });
    el.tglMeasNoise.addEventListener("change", ()=>{ app.plotDirty=true; });

    setupTabs();
    setupOrbitInteractions();
    setupPlotPanZoom();
  }

  function init(){
    buildStationBoard();
    bindUI();
    resetSim();
    syncFromUI();
    requestAnimationFrame(loop);
  }

  window.addEventListener("resize", ()=>{ app.plotDirty=true; app.viewDirty=true; });
  init();

})();
