"use strict";

import { Quaternion } from './Quaternion.js';
import { linspace, sliderChange, updateTextInput } from './utils.js';

const CANVAS = document.getElementById('canvas-q');
const thetaSlider = document.getElementById("thetaSlider-q");
const alphaSlider = document.getElementById("alphaSlider-q");
const betaSlider = document.getElementById("betaSlider-q");
const quaternionText = document.getElementById("quaternionText-q");
const resetButton = document.getElementById("resetButton-q");
const guidesCheckbox = document.getElementById("guidesCheckbox-q");


let lims = [-1.5, 1.5]
var layout = {
  margin: {
    l: 5, r: 5, b: 5, t: 25, pad: 4
  },
  scene:{
    aspectmode: "manual",
  aspectratio: {
    x: 1, y: 1, z: 1,
    },
  xaxis: {
    range: lims,
    dtick: 0.5,
  },
  yaxis: {
    range: lims,
    dtick: 0.5,
  },
  zaxis: {
    range: lims,
    dtick: 0.5,
  }},
};

let base = {
  x: [0, 0, 1/Math.sqrt(2)],
  y: [0, 0, 1/Math.sqrt(2)],
  z: [1, 0, 0],
}
let normal = [1, 0, 0];
const radius = 1;
const step = 0.01;
const stepPrecision = 2;
const angles = linspace(-3.2, step, 3.2);

const alpha_circle = {
  x: angles.map((a) => radius * Math.cos(a)),
  y: angles.map((a) => radius * Math.sin(a)),
  z:  new Array(angles.length).fill(0)
}

const beta_circle = {
  x: angles.map((a) => radius * Math.cos(a)),
  y: new Array(angles.length).fill(0),
  z: angles.map((a) => radius * Math.sin(a))
}

const theta_circle = {
  x: new Array(angles.length).fill(0),
  y: angles.map((a) => radius * Math.cos(a)),
  z: angles.map((a) => radius * Math.sin(a))
}


let normalData = {
  type: 'scatter3d',
  mode: 'lines+markers',
  x: [0, normal[0]],
  y: [0, normal[1]],
  z: [0, normal[2]],
  line: {
    width: 6,
    color: 'black',
  },
  marker: {
    size: 4,
    color: 'black',
  },
  name: 'normal',
}
let baseData =
  {
    type: 'scatter3d',
    mode: 'lines+markers',
    x: base.x,
    y: base.y,
    z: base.z,
    line: {
        width: 6,
        color: 'blue',
    },
    marker: {
        size: 4,
        color: 'blue',
        symbol: ['diamond', 'circle', 'circle'],
    },
    name :'base',
}
let alphaData = {
  type: 'scatter3d',
  mode: 'lines',
  x: [],
  y: [],
  z: [],
  line: {
      width: 3,
      color: 'green',
  },
  name: unescape('%u03B1'),
}
let betaData = {
  type: 'scatter3d',
  mode: 'lines',
  x: [],
  y: [],
  z: [],
  line: {
      width: 3,
      color: 'purple',
  },
  name: unescape('%u03B2'),
}
let thetaData = {
  type: 'scatter3d',
  mode: 'lines',
  x: [],
  y: [],
  z: [],
  line: {
      width: 3,
      color: 'orange',
  },
  name: unescape('%u03B8'),
}
var data = [baseData, normalData, alphaData, betaData, thetaData]
Plotly.newPlot(CANVAS, data, layout);


const updatePlot = () => {
  let theta = +thetaSlider.value;
  let alpha = +alphaSlider.value;
  let beta = +betaSlider.value;
  
  normal = [
      Math.cos(beta) * Math.cos(alpha), 
      Math.cos(beta) * Math.sin(alpha),
      Math.sin(beta)
  ]
  let sinHalfTheta = Math.sin(theta/2);
  let q = new Quaternion(
      Math.cos(theta/2),
      sinHalfTheta * normal[0],
      sinHalfTheta * normal[1],
      sinHalfTheta * normal[2],
  );
  quaternionText.innerText = "q = " + q.toString(3);

  let baseR = Quaternion.rotateMatrix([base.x, base.y, base.z], q)
  
  let alphaData = updateAlphaData(alpha);
  let betaData = updateBetaData(alpha, beta);
  let thetaData = updateThetaData(alpha, beta, theta);
  
  var newData = {
    x: [baseR[0], [0, normal[0]], alphaData.x, betaData.x, thetaData.x],
    y: [baseR[1], [0, normal[1]], alphaData.y, betaData.y, thetaData.y],
    z: [baseR[2], [0, normal[2]], alphaData.z, betaData.z, thetaData.z],
  }

  Plotly.restyle(CANVAS, newData);
}


const getArc = (circle, angle, angle0) => {
  let idx_0 = Math.floor((0 - angle0)/step)
  let idx_start, idx_end, arc;
  if (angle < 0){
    idx_start = Math.floor((angle - angle0)/step);
    idx_end = idx_0;
    arc = {
      x: circle.x.slice(idx_start, idx_end).concat([1, 0]),
      y: circle.y.slice(idx_start, idx_end).concat([0, 0]),
      z: circle.z.slice(idx_start, idx_end).concat([0, 0]),
    }
  }
  else{
    idx_start = idx_0;
    idx_end = Math.ceil((angle - angle0)/step) + 1;
    arc = {
      x: [0, 1].concat(circle.x.slice(idx_start, idx_end)),
      y: [0, 0].concat(circle.y.slice(idx_start, idx_end)),
      z: [0, 0].concat(circle.z.slice(idx_start, idx_end)),
    }
  }
  return arc;
}


const updateAlphaData = (alpha) => {
  return getArc(alpha_circle, alpha, angles[0]);
}


const updateBetaData = (alpha, beta) => {
  let arc = getArc(beta_circle, beta, angles[0])
  let q_alpha = new Quaternion(Math.cos(alpha/2), 0, 0, Math.sin(alpha/2));
  let mRot = Quaternion.rotateMatrix([arc.x, arc.y, arc.z], q_alpha);
  arc = {x: mRot[0], y: mRot[1], z:mRot[2]};
  return arc;
}


const updateThetaData = (alpha, beta, theta) => {
  let q_alpha = new Quaternion(Math.cos(alpha/2), 0, 0, Math.sin(alpha/2));
  let q_beta = new Quaternion(Math.cos(beta/2), 0, Math.sin(-beta/2), 0);
  let q = q_alpha.multiply(q_beta);
  //scale circle height and radius by beta
  let data = {
    x: new Array(theta_circle.x.length).fill(Math.sin(beta)),
    y: theta_circle.y.map((c) => Math.cos(beta) * c),
    z: theta_circle.z.map((c) => Math.cos(beta) * c),
  }
  let mRot = Quaternion.rotateMatrix([data.x, data.y, data.z], q);
  data = {x: mRot[0], y: mRot[1], z:mRot[2]};

  return data;
}


const reset = () => {
  thetaSlider.value = 0;
  alphaSlider.value = 0;
  betaSlider.value = 0;

  updateTextInput("thetaSliderText-q", 0, stepPrecision);
  updateTextInput("alphaSliderText-q", 0, stepPrecision);
  updateTextInput("betaSliderText-q", 0, stepPrecision);
  updatePlot();
  Plotly.relayout(CANVAS, {scene: {
    camera: {
      eye: {x: 1.00, y: 1.00, z:1.00}
    },
    ...layout.scene,
  }}
  );
}

const hideGuides = () => {
  let visible = guidesCheckbox.checked ? 'legendonly' : true;
  Plotly.restyle(CANVAS, {visible: visible}, [2, 3, 4]);
}

// oninput gives continous feedback but may not work in older versions of Internet Explorer
thetaSlider.oninput = () => sliderChange("thetaSlider-q", "thetaSliderText-q", stepPrecision, updatePlot);
alphaSlider.oninput = () => sliderChange("alphaSlider-q", "alphaSliderText-q", stepPrecision, updatePlot);
betaSlider.oninput = () => sliderChange("betaSlider-q", "betaSliderText-q", stepPrecision, updatePlot);
resetButton.onclick = reset;
guidesCheckbox.onclick = hideGuides;

reset();
