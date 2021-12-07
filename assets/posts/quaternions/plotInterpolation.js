"use strict";

import { Quaternion } from './Quaternion.js';
import { sliderChange, updateTextInput} from './utils.js'

const CANVAS = document.getElementById('canvas');
const psiStartNumber = document.getElementById("psiStartNumber");
const thetaStartNumber = document.getElementById("thetaStartNumber");
const phiStartNumber = document.getElementById("phiStartNumber");
const psiEndNumber = document.getElementById("psiEndNumber");
const thetaEndNumber = document.getElementById("thetaEndNumber");
const phiEndNumber = document.getElementById("phiEndNumber");
const tSlider = document.getElementById("tSlider");
const resetButton = document.getElementById("resetButton");
const interpRadios = Array.from(document.getElementsByName("interpRadios"))
const defaultButton = document.getElementById("defaultButton");

const sliderPrecision = 2;
const RAD_PER_DEG = Math.PI / 180;

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

//aeroplane
let hstab = 0.8/2
let tail  = 0.4;
let wing  = 1.8/2;
let body =  2/2;
let wing_y = 0;
let base = {
  x: [-hstab, hstab,  0,        0, 0,      wing,    -wing,       0,    0],
  y: [-body, -body, -body, -body,  wing_y, wing_y,  wing_y, wing_y, body],
  z: [ tail,  tail,  tail,     0,  0,      0,         0,      0,       0],
}


let baseData =
  {
    type: 'scatter3d',
    mode: 'lines+markers',
    x: base.x,
    y: base.y,
    z: base.z,
    line: {
        width: 12,
        color: 'blue',
    },
    marker: {
        size: 3,
        color: 'blue',
        symbol: 'circle',
    },
    name :'initial',
}
let destData =
  {
    type: 'scatter3d',
    mode: 'lines+markers',
    x: base.x,
    y: base.y,
    z: base.z,
    line: {
        width: 12,
        color: 'blue',
    },
    marker: {
        size: 3,
        color: 'blue',
        symbol: 'circle',
    },
    name :'destination',
    opacity: 0.2,
}

var data = [baseData, destData]
Plotly.newPlot(CANVAS, data, layout);

let interpMethod = Quaternion.lerp;
const chooseInterpMethod = () => {
  if (interpRadios.find(x => x.id == "lerpRadio").checked){
    interpMethod = Quaternion.lerp;
  }
  else{
    interpMethod = Quaternion.slerp;
  }
  updatePlot();
}
interpRadios.forEach(node => node.onclick = chooseInterpMethod);


const getEulerQuaternion = (psi, theta, phi) => {
  let qPsi = new Quaternion(Math.cos(psi/2), 0, 0, Math.sin(psi/2));
  let qTheta = new Quaternion(Math.cos(theta/2), Math.sin(theta/2), 0, 0);
  let qPhi = new Quaternion(Math.cos(phi/2), 0, Math.sin(phi/2), 0);
  let qR = qPsi.multiply(qTheta).multiply(qPhi);
  return qR;
}


const updatePlot = () => {
  let qStart, qEnd, qt;
  let psi, theta, phi;

  psi = +psiEndNumber.value * RAD_PER_DEG ;
  theta = +thetaEndNumber.value * RAD_PER_DEG ;
  phi = +phiEndNumber.value * RAD_PER_DEG ;
  
  qEnd = getEulerQuaternion(psi, theta, phi);
  let destRot = Quaternion.rotateMatrix([base.x, base.y, base.z], qEnd);

  psi = +psiStartNumber.value * RAD_PER_DEG ;
  theta = +thetaStartNumber.value * RAD_PER_DEG ;
  phi = +phiStartNumber.value * RAD_PER_DEG ;

  qStart = getEulerQuaternion(psi, theta, phi);
  qt = interpMethod(qStart, qEnd, +tSlider.value);
  let baseRot = Quaternion.rotateMatrix([base.x, base.y, base.z], qt)

  var newData = {
    x: [baseRot[0], destRot[0]],
    y: [baseRot[1], destRot[1]],
    z: [baseRot[2], destRot[2]],
  }

  Plotly.restyle(CANVAS, newData);
}

const numberInputChange = (id) => {
  let element = document.getElementById(id);
  if (+element.value > +element.max) element.value = element.max;
  if (+element.value < +element.min) element.value = element.min;
  updatePlot();
}


const setDefault = () => {
  psiStartNumber.value = -10;
  thetaStartNumber.value = 10;
  phiStartNumber.value = 30;

  psiEndNumber.value = -100;
  thetaEndNumber.value = -10;
  phiEndNumber.value = -45;

  updatePlot();
}


const reset = () => {
  psiStartNumber.value = 0;
  thetaStartNumber.value = 0;
  phiStartNumber.value = 0;

  psiEndNumber.value = 0;
  thetaEndNumber.value = 0;
  phiEndNumber.value = -0;

  tSlider.value = 0;
  updateTextInput("tSliderText", 0, sliderPrecision);

  updatePlot();
  Plotly.relayout(CANVAS, {scene: {
    camera: {
      eye: {x: 1.25, y: 1.25, z:1.25} 
    },
    ...layout.scene,
  }}
  );
}


// oninput gives continous feedback but may not work in older versions of Internet Explorer
phiStartNumber.oninput = () => numberInputChange("phiStartNumber");
thetaStartNumber.oninput = () => numberInputChange("thetaStartNumber");
psiStartNumber.oninput = () => numberInputChange("psiStartNumber");
phiEndNumber.oninput = () => numberInputChange("phiEndNumber");
thetaEndNumber.oninput = () => numberInputChange("thetaEndNumber");
psiEndNumber.oninput = () => numberInputChange("psiEndNumber");
tSlider.oninput = () => sliderChange("tSlider", "tSliderText", sliderPrecision, updatePlot)
resetButton.onclick = reset;
defaultButton.onclick = setDefault;

reset();
