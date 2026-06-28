import data from './lengths_1880-2025.json' with { type: 'json' };
import { genderColors, colors } from './constants.js';

const CANVAS = document.getElementById('canvas-lengths');

let countData = []
for (const gender of ['M', 'F']) {
    // unique names
    countData.push({
        x: data['lengths'],
        y: data[gender]['count names'],
        type: 'bar',
        name: gender,
        marker: {
            color: genderColors[gender],
        }
    })
}

var layout1 = {
    title: {text: 'Unique Names in the USA (1880-2025)'},
    xaxis: {title: {text: 'Length'}, tickvals: data['lengths']},
    yaxis: {title: {text: 'Count'}, range: [0, null]},
};
Plotly.newPlot(CANVAS, countData, layout1);


