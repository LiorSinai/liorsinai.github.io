import data from './top_names.json' with { type: 'json' };
import { genderColors, colors } from './constants.js';

// const data = require('./output/top_names.json');

const CANVAS = 'canvas-top-names';
const CANVAS_BAR = 'canvas-top-names-bar';

for (const gender of ['M', 'F']) {
    let i = -1;
    let names = Object.keys(data[gender]);
    let firstYear = names.map(key => data[gender][key]['years at top'][0])
    let order = firstYear.map((val, idx) => ({ val, idx })).sort((a, b) => a.val - b.val)
    let xaxis = gender == 'M' ? 'x1' : 'x2'
    let yaxis = gender == 'M' ? 'y1' : 'y2'
    
    // Counts
    let sortedNames = order.map(({ idx }) => names[idx])
    let count_data = []
    for (const name of sortedNames) {
        i++;
        count_data.push({
            x: data.years,
            y: data[gender][name]['count'],
            type: 'line',
            name: name,
            line: {dash: 'solid', color: colors[i]},
            xaxis: xaxis, yaxis: yaxis,
        })
    }

    var layout = {
        title: {text: `#1 Names in the USA (${gender})`},
        xaxis: {title: {text: 'Year'}},
        yaxis: {title: {text: 'Count'}, range: [0, null]},
    };
    Plotly.newPlot(`${CANVAS}-${gender}`, count_data, layout);
    
    // Frequency
    let frequency = names.map(key => data[gender][key]['years at top'].length)
    let freqData = [{
        x: order.map(({ idx }) => names[idx]),
        y: order.map(({ idx }) => frequency[idx]),
        type: 'bar',
        name: name,
        marker: {
            color: genderColors[gender],
        }
    }]
    var layout2 = {
        title: {text: `#1 Names in the USA (${gender})`},
        xaxis: {title: {text: 'Names'}},
        yaxis: {title: {text: 'Years Ranked #1'}, range: [0, 80]},
    };
    Plotly.newPlot(`${CANVAS_BAR}-${gender}`, freqData, layout2);
}

