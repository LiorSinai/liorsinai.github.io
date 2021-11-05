import { Matrix, rotateWithR, getTaitBryanZYX } from './EulerAngles.js';
import { linspace, sliderChange, updateTextInput } from './utils.js';

const CANVAS = document.getElementById('canvas-tb');
const phiSlider = document.getElementById("phiSlider-tb");
const theteSlider = document.getElementById("thetaSlider-tb");
const psiSlider = document.getElementById("psiSlider-tb");
const resetButton = document.getElementById("resetButton-tb");
const guidesCheckbox = document.getElementById("guidesCheckbox-tb");


let lims = [-1.5, 1.5]
var layout = {
  margin: {
    l: 40, r: 40, b: 40, t: 40, pad: 4
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
  x: [0, 0, 0, 0, 1],
  y: [0, 0, 1, 0, 0],
  z: [1, 0, 0, 0, 0],
}
const radius = 1;
const step = 0.01;
const stepPrecision = 2;
const angles = linspace(-3.2, step, 3.2);

const psiCircle = {
    x: angles.map((a) => radius * Math.cos(a)),
    y: angles.map((a) => radius * Math.sin(a)),
    z: new Array(angles.length).fill(0),
}

const thetaCircle = {
    x: angles.map((a) => radius * Math.cos(a)),
    y: new Array(angles.length).fill(0),
    z: angles.map((a) => radius * Math.sin(a)),
}

const phiCircle = {
    x: new Array(angles.length).fill(0),
    y: angles.map((a) => radius * Math.cos(a)),
    z: angles.map((a) => radius * Math.sin(a)),
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
        size: 3.5,
        color: 'blue',
        symbol: 'circle',
    },
    name :'base',
}
let psiData = {
    type: 'scatter3d',
    mode: 'lines',
    x: psiCircle.x,
    y: psiCircle.y,
    z: psiCircle.z,
    line: {
        width: 3,
        color: 'orange',
    },
    name: unescape('%u03C8'),
  }
let thetaData = {
    type: 'scatter3d',
    mode: 'lines',
    x: thetaCircle.x,
    y: thetaCircle.y,
    z: thetaCircle.z,
    line: {
        width: 3,
        color: 'green',
    },
    name: unescape('%u03B8'),
  }
let phiData = {
    type: 'scatter3d',
    mode: 'lines',
    x: phiCircle.x,
    y: phiCircle.y,
    z: phiCircle.z,
    line: {
        width: 3,
        color: 'purple',
    },
    name: unescape('%u03C6'),
  }

var data = [baseData, psiData, thetaData, phiData]
Plotly.newPlot(CANVAS, data, layout);


const updatePlot = () => {
    let phi = +phiSlider.value;
    let theta = +theteSlider.value;
    let psi = +psiSlider.value;

    let R = getTaitBryanZYX(phi, theta, psi);
    let vR = R.multiply(new Matrix([base.x, base.y, base.z]));

    let psiR = R.multiply(new Matrix([psiCircle.x, psiCircle.y, psiCircle.z]));
    let thetaR = R.multiply(new Matrix([thetaCircle.x, thetaCircle.y, thetaCircle.z]));
    let phiR = R.multiply(new Matrix([phiCircle.x, phiCircle.y, phiCircle.z]));

    var newData = {
        x: [vR.index(0), psiR.index(0), thetaR.index(0), phiR.index(0)],
        y: [vR.index(1), psiR.index(1), thetaR.index(1), phiR.index(1)],
        z: [vR.index(2), psiR.index(2), thetaR.index(2), phiR.index(2)],
      }
    
    Plotly.restyle(CANVAS, newData);
}


const reset = () => {
    phiSlider.value = 0;
    theteSlider.value = 0;
    psiSlider.value = 0;

    updateTextInput("phiSliderText-tb", 0, stepPrecision);
    updateTextInput("thetaSliderText-tb", 0, stepPrecision);
    updateTextInput("psiSliderText-tb", 0, stepPrecision);
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
  Plotly.restyle(CANVAS, {visible: visible}, [1, 2, 3]);
}

// oninput gives continous feedback but may not work in older versions of Internet Explorer
phiSlider.oninput = () => sliderChange("phiSlider-tb", "phiSliderText-tb", stepPrecision, updatePlot);
theteSlider.oninput = () => sliderChange("thetaSlider-tb", "thetaSliderText-tb", stepPrecision, updatePlot);
psiSlider.oninput = () => sliderChange("psiSlider-tb", "psiSliderText-tb", stepPrecision, updatePlot);
resetButton.onclick = reset;
guidesCheckbox.onclick = hideGuides;

reset();
