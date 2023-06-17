"use strict";

import { normalRandom, bivariateFromIndependentNormal } from './normal.js';
import { wassersteinMetric } from './wasserstein.js';

let CANVAS = document.getElementById('canvas-bivariate');
let precision = 2;

/* initial plot */ 

let data1 = {numSamples: 100};
let data2 = {numSamples: 100};

data1.z1 = Array.from({length: data1.numSamples}, () => normalRandom());
data1.z2 = Array.from({length: data1.numSamples}, () => normalRandom());
data2.z1 = Array.from({length: data2.numSamples}, () => normalRandom());
data2.z2 = Array.from({length: data2.numSamples}, () => normalRandom());

let defaults = {
  meanX: 0.0,
  meanY: 0.0,
  stdX: 1.0,
  stdY: 1.0,
  corr: 0.0,
}

let descriptors1 = {...defaults}
let xy = bivariateFromIndependentNormal(
  data1.z1,
  data1.z2, 
  [descriptors1.meanX, descriptors1.meanY], 
  [descriptors1.stdX, descriptors1.stdY], 
  descriptors1.corr
  );
data1.x = xy.x
data1.y = xy.y

let descriptors2 = {...defaults}

xy = bivariateFromIndependentNormal(
  data2.z1,
  data2.z2, 
  [descriptors2.meanX, descriptors2.meanY], 
  [descriptors2.stdX, descriptors2.stdY], 
  descriptors2.corr
  );
data2.x = xy.x
data2.y = xy.y

let limits = [-20.0, 20.0]
let layout = {
  //autosize: true,
  xaxis: {
    range: limits,
    scaleanchor: 'y',
    automargin: true,
  },
  yaxis: {
    range: limits,
    automargin: true,
    constrain: 'domain',
  },
  margin: {
    t: 60,
    l: 40,
    r: 40,
    b: 40,
    }
};
let config = {responsive: true};

let trace1 = {
  x: data1.x,
  y: data1.y,
  mode: 'markers',
  type: 'scatter',
  name: '1',
};
let trace2 = {
  x: data2.x,
  y: data2.y,
  mode: 'markers',
  type: 'scatter',
  name: '2',
};

let plotData = [trace1, trace2];

Plotly.newPlot(CANVAS, plotData, layout, config);

const updateSimilarity = () => {
  let W2 = wassersteinMetric(data1.x, data1.y, data2.x, data2.y);
  let W2span = document.getElementById("similarity");
  W2span.innerHTML = W2.toFixed(2);
}
updateSimilarity();

/* selected */
let selectedDescriptors = descriptors1;
let selectedData = data1;

const updateSelectedData = () => {
  selectedData.z1 = Array.from({length: selectedData.numSamples}, () => normalRandom());
  selectedData.z2 = Array.from({length: selectedData.numSamples}, () => normalRandom());
}

/** sliders ****/
let meanXSlider =  document.getElementById('meanXSlider');
let meanXSliderText = document.getElementById('meanXSliderText')
let meanYSlider =  document.getElementById('meanYSlider');
let meanYSliderText = document.getElementById('meanYSliderText')
let stdXSlider =  document.getElementById('stdXSlider');
let stdXSliderText = document.getElementById('stdXSliderText')
let stdYSlider =  document.getElementById('stdYSlider');
let stdYSliderText = document.getElementById('stdYSliderText')
let corrSlider =  document.getElementById('corrSlider');
let corrSliderText = document.getElementById('corrSliderText')

let sliders = [
  [meanXSlider, meanXSliderText, 'meanX'], 
  [meanYSlider, meanYSliderText, 'meanY'], 
  [stdXSlider, stdXSliderText, 'stdX'],
  [stdYSlider, stdYSliderText, 'stdY'],
  [corrSlider, corrSliderText, 'corr']
];

sliders.forEach(args => {
  let slider = args[0];
  let sliderText = args[1];
  let property = args[2];
  slider.oninput = () => updateAllFromSlider(slider, sliderText, property);
})

const updateTextInput = (element, value, precision) => {
  let numSpaces;
  if (value <= -10){numSpaces = 0}
  else if (value < 0){ numSpaces = 1}
  else if (value < 10){ numSpaces = 2}
  else {numSpaces = 1};
  let spaces = ' '.repeat(numSpaces);
  element.innerText = (+value).toFixed(precision); 
}

const updateAllFromSlider = (slider, sliderText, property) => {
  updateTextInput(sliderText, +slider.value, precision);
  selectedDescriptors[property] = +slider.value;
  updatePlot();
  updateSimilarity();
}

const updateSliderTexts = () => {
  sliders.forEach(args => {
    let slider = args[0];
    let sliderText = args[1];
    updateTextInput(sliderText, +slider.value, precision);
  })
}

const randomInRange = (min, max) => Math.random() * (max - min) + min;

const updateSlidersAndDescriptorsRandom = () => {
  sliders.forEach(args => {
    let slider = args[0];
    let sliderText = args[1];
    let property = args[2];
    let value = randomInRange(+slider.min, +slider.max);
    slider.value = value;
    selectedDescriptors[property] = value;
    updateTextInput(sliderText, +slider.value, precision);
  })
}

/* radio buttons */
let radioData1 = document.getElementById("radio-data-1");
let radioData2 = document.getElementById("radio-data-2");

radioData1.oninput = () => updateSelectedAndSliders();
radioData2.oninput = () => updateSelectedAndSliders();

const updateSelectedAndSliders = () => {
  selectedDescriptors = radioData1.checked ? descriptors1 : descriptors2;
  selectedData = radioData1.checked ? data1 : data2;
  meanXSlider.value = selectedDescriptors.meanX;
  meanYSlider.value = selectedDescriptors.meanY;
  stdXSlider.value = selectedDescriptors.stdX;
  stdYSlider.value = selectedDescriptors.stdY;
  corrSlider.value = selectedDescriptors.corr;
  updateSliderTexts();
}

/* buttons */
const updateOutputs = () => {
  updateSelectedData();
  updatePlot();
  updateSimilarity();
}

let buttonRefresh = document.getElementById('buttonRefresh')
buttonRefresh.onclick = () => {
  selectedData.z1 = Array.from({length: selectedData.numSamples}, () => normalRandom());
  selectedData.z2 = Array.from({length: selectedData.numSamples}, () => normalRandom());
  updateOutputs();
}

let buttonReset = document.getElementById('buttonReset')
let formControls = document.getElementById('controls')
buttonReset.onclick = () => {
  let firstIsChecked = radioData1.checked;
  formControls.reset();
  if (!firstIsChecked) {radioData2.checked = true}  // undo reset
  updateSliderTexts();
  Object.assign(selectedDescriptors, defaults);
  updateOutputs();
}

let buttonRandom = document.getElementById('buttonRandomise')
buttonRandom.onclick = () => {
  updateSlidersAndDescriptorsRandom();
  updateOutputs();
}

/* update plot */
const updatePlot = () => {
  xy = bivariateFromIndependentNormal(
    data1.z1,
    data1.z2, 
    [descriptors1.meanX, descriptors1.meanY], 
    [descriptors1.stdX, descriptors1.stdY], 
    descriptors1.corr
    );
  data1.x = xy.x
  data1.y = xy.y

  xy = bivariateFromIndependentNormal(
    data2.z1,
    data2.z2, 
    [descriptors2.meanX, descriptors2.meanY], 
    [descriptors2.stdX, descriptors2.stdY], 
    descriptors2.corr
    );
  data2.x = xy.x
  data2.y = xy.y

  let update = {
    x: [data1.x, data2.x],
    y: [data1.y, data2.y],
  }

  Plotly.restyle(CANVAS, update)
}
