import data from './lior.json' with { type: 'json' };
import nameData from './name_rank_quantiles.json' with { type: 'json' };
import { genderColors, colors } from './constants.js';
// const data = require('./output/lior.json');

const CANVAS = document.getElementById('canvas-my-name');

let i = -1;
let countData = []
let rankData = []
for (const gender of ['M', 'F']){
    let dash = gender == 'M' ? 'solid' : 'dash'
    rankData.push({
        x: nameData.years,
        y: nameData[gender]['max dense rank'],
        type: 'line',
        name: `Lowest dense rank (${gender})`,
        line: {dash: dash, color: 'rgba(0, 0, 0, 0.6)'},
        xaxis: 'x2', yaxis: 'y2',
    })
}
for (const gender of ['M', 'F']) {
    let names = Object.keys(data[gender]);
    for (const name of names) {
        i++;
        let label = `${name} (${gender})`
        countData.push({
            x: data.years,
            y: data[gender][name]["count"],
            type: 'line',
            name: label,
            line: {dash: 'solid', color: colors[i]},
            legendgroup: label, showlegend: true,
        })
        rankData.push({
            x: data.years,
            y: data[gender][name]["rank"],
            type: 'line',
            name: label,
            line: {dash: 'solid', color: colors[i]},
            xaxis: 'x2', yaxis: 'y2',
            legendgroup: label, showlegend: false,
        })
    }
}

var isMobile = window.innerWidth <= 768;
var layout = {
    title: {text: `Name Frequency in the USA`},
    xaxis: {title: {text: 'Year'}},
    yaxis: {title: {text: 'Count'}, range: [0, null]},
    xaxis2: {title: {text: 'Year'}},
    yaxis2: {title: {text: 'Dense Rank'}, range: [0, 1000]},
    grid: { 
        rows: isMobile ? 2 : 1,
        columns: isMobile ? 1 : 2,
        pattern: 'independent'
    },
};
Plotly.newPlot(CANVAS, [...countData, ...rankData], layout);