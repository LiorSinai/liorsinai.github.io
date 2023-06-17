import {Matrix, matrixAdd, matrixMultiply} from './linearAlgebra.js';

const mean = (array) => {
    let result = 0;
    for (let i=0; i <array.length; i++){
        result += array[i];
    }
    result /= array.length;
    return result;
}

const variance = (array, mean) => {
    let result = 0;
    for (let i=0; i <array.length; i++){
        result += (array[i] - mean) * (array[i] - mean);
    }
    result /= (array.length - 1);
    return result;
}

const covariance = (array1, array2, mean1, mean2) => {
    if (array1.length != array2.length) {throw `unequal lengths: ${array1.length}!=${array2.length}`};
    let result = 0;
    for (let i=0; i <array1.length; i++){
        result += (array1[i] - mean1) * (array2[i] - mean2);
    }
    result /= (array1.length - 1);
    return result;
}

const bivariateStatistics = (x, y) => {
    let stats = {}
    stats.meanX = mean(x);
    stats.meanY = mean(y);
    stats.varX = variance(x, stats.meanX);
    stats.varY = variance(y, stats.meanY);
    stats.cov = covariance(x, y, stats.meanX, stats.meanY);
    return stats;
}

const wassersteinMetric = (x1, y1, x2, y2) => {
    let stats1 = bivariateStatistics(x1, y1);
    let stats2 = bivariateStatistics(x2, y2);

    let meanDiff = [stats1.meanX - stats2.meanX, stats1.meanY - stats2.meanY];
    let meanDistance = meanDiff[0] * meanDiff[0] + meanDiff[1] * meanDiff[1];

    let C1 = new Matrix([[stats1.varX, stats1.cov], [stats1.cov, stats1.varY]]);
    let C2 = new Matrix([[stats2.varX, stats2.cov], [stats2.cov, stats2.varY]]);
    let C2sqrt = C2.sqrt();
    let meanCovSquare = matrixMultiply(C2sqrt, matrixMultiply(C1, C2sqrt));
    let meanCov = meanCovSquare.sqrt();
    let covDistance = matrixAdd(C1, matrixAdd(C2, meanCov.multiplyScalar(-2))).trace();

    let distance = meanDistance + covDistance;
    return distance;
}

export {
    mean,
    variance,
    covariance,
    wassersteinMetric,
}