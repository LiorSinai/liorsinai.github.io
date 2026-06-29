import data from './name_rank_quantiles.json' with { type: 'json' };
import { genderColors, colors } from './constants.js';

const CANVAS_RATIO = document.getElementById('canvas-name-ratios');
const CANVAS_COUNTS = document.getElementById('canvas-name-counts');
//const CANVAS_CHANGE = document.getElementById('canvas-name-change');

let countData = []
let fractionData = []
let ratioData = []
let changeData = []
for (const gender of ['M', 'F']) {
    let i = -1;
    let dash =  gender == 'M' ? 'solid' : 'dash'
    // counts
    let unique_count = data[gender]['1.0']
    for (const q of ['0.25', '0.5', '0.75', '1.0']){
        i++;
        let label = `${q*100}% of ${gender}`
        countData.push({
            x: data.years,
            y: data[gender][q],
            type: 'line',
            name: label,
            line: {dash: dash, color: colors[i]},
            legendgroup: label, showlegend: true,
        })
        let fractions = data[gender][q].slice();
        for (let j = 0; j < fractions.length; j++) {
            fractions[j] = fractions[j] / unique_count[j] * 100
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

    // dense rank
    // countData.push({
    //     x: data.years,
    //     y: data[gender]['max dense rank'],
    //     type: 'line',
    //     name: `${gender} dense rank`,
    //     line: {dash: 'dash', color: colors[gender]},
    // })

    //ratios
    let total_count = data[gender]['total births']
    const ratios = total_count.map((num, index) => num / unique_count[index]);
    ratioData.push({
        x: data.years,
        y: ratios,
        type: 'line',
        name: `${gender}`,
        line: {dash: 'solid', color: genderColors[gender]},
    })

    // changeData.push({
    //     //x: data.years,
    //     y: data[gender]['change'],
    //     type: 'bar',
    //     name: `${gender} change`,
    //     marker: {color: genderColors[gender]},
    // })
}

var isMobile = window.innerWidth <= 768;
var layout1 = {
    title: {text: 'Unique Names per Year in the USA'},
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
Plotly.newPlot(CANVAS_COUNTS, [...countData, ...fractionData], layout1);

var layout2 = {
    title: {text: 'Unique Names in the USA'},
    xaxis: {title: {text: 'Year'}},
    yaxis: {title: {text: 'Births / Name'}, range: [0, null]},
};
Plotly.newPlot(CANVAS_RATIO, ratioData, layout2);

//Plotly.newPlot(CANVAS_CHANGE, changeData);