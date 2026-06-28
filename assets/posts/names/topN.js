import data from './topN.json' with { type: 'json' };
import { colors } from './constants.js';
// const data = require('./output/topN.json');

const CANVAS = document.getElementById('canvas-topN');

let thresholds = Object.keys(data['M']);
let genders = ['M', 'F'];

// Count data
let countData = []
let fractionData = []
for (const gender of genders) {
    let i = -1;
    let dash =  gender == 'M' ? 'solid' : 'dash';
    let totals = data[gender]['all']['counts']
    for (const t of thresholds) {
        i++;
        let label = !isNaN(Number(t)) ? `Top ${t} ${gender}` : `${t} ${gender}`;
        countData.push({
            x: data.years,
            y: data[gender][t]['counts'],
            type: 'line',
            name: label,
            line: {dash: dash, color: colors[i]},
            legendgroup: label, showlegend: true,
        })

        let fractions = data[gender][t]['counts'].slice();
        for (let j = 0; j < fractions.length; j++) {
            fractions[j] = fractions[j] / totals[j] * 100
        }
        fractionData.push({
            x: data.years,
            y: fractions,
            type: 'line',
            name: label,
            line: {dash: dash, color: colors[i]},
            xaxis: 'x2', yaxis: 'y2',
            legendgroup: label, showlegend: false,
        })
    }
}

var isMobile = window.innerWidth <= 768;
var layout = {
    title: {text: 'Births by Name Ranking in the USA'},
    xaxis: {title: {text: 'Year'}},
    yaxis: {title: {text: 'Count'}, range: [0, null]},
    xaxis2: {title: {text: 'Year'}},
    yaxis2: {title: {text: 'Proportion (%)'}, range: [0, null]},
    grid: { 
        rows: isMobile ? 2 : 1,
        columns: isMobile ? 1 : 2,
        pattern: 'independent'
    },
};
Plotly.newPlot(CANVAS, [...countData, ...fractionData], layout);

