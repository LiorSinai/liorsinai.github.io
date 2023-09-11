compositionCanvas = document.getElementById('plot-composition');

var data = [{
    values: [
        0.49270276567601434, 
        0.3634221566065064,
        0.11092827454559061,
        0.01832728014231497,
        0.007395218303039374,
        0.004072728920514438,
        0.0031515758060198795
    ],
    labels: [
        'none',
        '1 pair',
        '2 pairs',
        '3 pairs',
        '1 triple',
        '1 pair, 1 triple',
        'rest'
    ],
    type: 'pie'
}];
  
var layout = {
    margin: {
        l: 5, r: 5, b: 5, t: 5, pad: 0
      },
    legend: {
        x: 1.8,
        xanchor: 'right',
        y: 0.75
    }
    // height: 450,
    // width: 450
};
Plotly.newPlot(compositionCanvas, data, layout);