"use strict";

import { addArrays, scalarMultiplyArray } from './arrays.js';

// Standard Normal distribution using Box-Muller transform.
const normalRandom = (mean=0, stdev=1) => {
    let u = 1 - Math.random(); // Converting [0,1) to (0,1]
    let v = Math.random();
    let z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    return z * stdev + mean;
}

const bivariateNormal = (mean, std, corr, numSamples) => {
    let z1 = Array.from({length: numSamples}, () => normalRandom());
    let z2 = Array.from({length: numSamples}, () => normalRandom());

    return bivariateFromIndependentNormal(z1, z2, mean, std, corr)
}

const bivariateFromIndependentNormal = (z1, z2, mean, std, corr) => {
    let x0 = z1;
    let x = scalarMultiplyArray(x0, std[0]).map(v => v + mean[0]);
    let y0 = addArrays(
        scalarMultiplyArray(z1, corr),  scalarMultiplyArray(z2, Math.sqrt(1 - corr * corr))
    )
    let y = scalarMultiplyArray(y0, std[1]).map(v => v + mean[1]);
    return {x, y}
}

export{ 
    normalRandom, 
    bivariateNormal,
    bivariateFromIndependentNormal,
}