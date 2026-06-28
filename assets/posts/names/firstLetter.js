import data from './first_letter_1880-2025.json' with { type: 'json' };
// const data = require('./output/first_letter_1880-2025.json');
import { genderColors, colors } from './constants.js';

const CANVAS_NAMES = document.getElementById('canvas-first-letter-names');
const CANVAS_TOTAL = document.getElementById('canvas-first-letter-counts');

let nameData = []
let totalData = []
for (const gender of ['M', 'F']) {
    // unique names
    nameData.push({
        x: data['letters'],
        y: data[gender]['count names'],
        type: 'bar',
        name: gender,
        marker: {
            color: genderColors[gender],
        }
    })
    // totals
    totalData.push({
        x: data['letters'],
        y: data[gender]['count totals'],
        type: 'bar',
        name: gender,
        marker: {
            color: genderColors[gender],
        }
    })
}

var layout1 = {
    title: {text: 'Unique Names in the USA (1880-2025)'},
    xaxis: {title: {text: 'First Letter'}},
    yaxis: {title: {text: 'Count'}, range: [0, null]},
};
Plotly.newPlot(CANVAS_NAMES, nameData, layout1);

var layout2 = {
    title: {text: 'Total Name Counts in the USA (1880-2025)'},
    xaxis: {title: {text: 'First Letter'}},
    yaxis: {title: {text: 'Count'}, range: [0, null]},
};
Plotly.newPlot(CANVAS_TOTAL, totalData, layout2);


