
"use strict";

export const linspace = (start, step, stop) => {
    if (stop < start) throw stop + "<" + start + " ; stop must be greater than start" 
    let len = Math.floor((stop - start)/step) + 1;
    let arr = new Array(len).fill(0);
    arr[0] = start;
    for (var i = 1; i < len; i ++){
        arr[i] = arr[i - 1] + step;
    }
    return arr;
}

export const norm = (array) => {
    let mag2 = array.map(a => a**2).reduce((a,b) => a + b, 0)
    return Math.sqrt(mag2);
}

export const sliderChange = (sliderID, textID, precision, callback) => {
    var value = document.getElementById(sliderID).value;
    updateTextInput(textID, value, precision);
    callback();
  }
  
export const updateTextInput = (textID, value, precision) => {
  document.getElementById(textID).innerText = (+value).toFixed(precision); 
  }
  

