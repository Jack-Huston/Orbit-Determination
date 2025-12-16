
"use strict";

/* ================= CONSTANTS ================= */
const MU = 398600;
const R_E = 6378;
const OMEGA_E = 2*Math.PI/86400;
const DT = 10;
const N_STATIONS = 12;

/* ================= CANVAS ================= */
const canvas = document.getElementById("orbitCanvas");
const ctx = canvas.getContext("2d");

function resize() {
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}
window.addEventListener("resize", resize);
resize();

/* ================= UTIL ================= */
const norm = (x,y)=>Math.hypot(x,y);
const wrap = a=>Math.atan2(Math.sin(a),Math.cos(a));

/* ================= DYNAMICS ================= */
function dynamics(x) {
  const r = norm(x[0],x[2]);
  return [
    x[1],
    -MU*x[0]/(r**3),
    x[3],
    -MU*x[2]/(r**3)
  ];
}

function rk4(x) {
  const k1=dynamics(x);
  const k2=dynamics(x.map((v,i)=>v+0.5*DT*k1[i]));
  const k3=dynamics(x.map((v,i)=>v+0.5*DT*k2[i]));
  const k4=dynamics(x.map((v,i)=>v+DT*k3[i]));
  return x.map((v,i)=>v+DT/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
}

/* ================= STATIONS ================= */
function station(i,t) {
  const th=i*Math.PI/6+OMEGA_E*t;
  return {
    x:R_E*Math.cos(th),
    y:R_E*Math.sin(th),
    xd:-OMEGA_E*R_E*Math.sin(th),
    yd: OMEGA_E*R_E*Math.cos(th)
  };
}

function visible(x,s) {
  const rx=x[0]-s.x, ry=x[2]-s.y;
  return rx*s.x+ry*s.y>0;
}

/* ================= EKF ================= */
function A_jac(x){
  const r=Math.hypot(x[0],x[2]);
  const r5=r**5;
  return [
    [0,1,0,0],
    [MU*(2*x[0]**2-x[2]**2)/r5,0,3*MU*x[0]*x[2]/r5,0],
    [0,0,0,1],
    [3*MU*x[0]*x[2]/r5,0,MU*(2*x[2]**2-x[0]**2)/r5,0]
  ];
}

/* ================= STATE ================= */
let xTrue, xHat, P, t;
let running=false;

/* ================= DRAW ================= */
function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(canvas.width/2,canvas.height/2);
  const scale=canvas.width/(2*9000);

  ctx.fillStyle="#22c55e";
  ctx.beginPath();
  ctx.arc(0,0,R_E*scale,0,2*Math.PI);
  ctx.fill();

  ctx.fillStyle="#f97316";
  ctx.beginPath();
  ctx.arc(xTrue[0]*scale,-xTrue[2]*scale,4,0,2*Math.PI);
  ctx.fill();

  ctx.fillStyle="#3b82f6";
  ctx.beginPath();
  ctx.arc(xHat[0]*scale,-xHat[2]*scale,4,0,2*Math.PI);
  ctx.fill();

  for(let i=0;i<N_STATIONS;i++){
    const s=station(i,t);
    ctx.fillStyle="#a855f7";
    ctx.beginPath();
    ctx.arc(s.x*scale,-s.y*scale,3,0,2*Math.PI);
    ctx.fill();

    if(visible(xTrue,s)){
      ctx.strokeStyle="#ef4444";
      ctx.beginPath();
      ctx.moveTo(xTrue[0]*scale,-xTrue[2]*scale);
      ctx.lineTo(s.x*scale,-s.y*scale);
      ctx.stroke();
    }
  }

  ctx.restore();
}

/* ================= LOOP ================= */
function step(){
  if(!running) return;
  xTrue=rk4(xTrue);
  xHat=rk4(xHat);
  t+=DT;
  draw();
  requestAnimationFrame(step);
}

/* ================= CONTROLS ================= */
function reset(){
  const alt=Number(altitude.value);
  const r0=R_E+alt;
  const v0=Math.sqrt(MU/r0);
  xTrue=[r0,0,0,v0];
  xHat=[...xTrue];
  t=0;
  draw();
}

startBtn.onclick=()=>{running=true;step();};
resetBtn.onclick=()=>{running=false;reset();};

reset();
