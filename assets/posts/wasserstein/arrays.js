const arrayRange = (start, stop, step) => {
    return Array.from(
    { length: Math.ceil((stop - start) / step) + 1 },
    (value, index) => start + index * step
    );
}

const sumArray = (array) => 
  array.reduce((partialSum, a) => partialSum + a, 0);
const scalarMultiplyArray = (array, s) => array.map(x => x * s)

const addArrays = (array1, array2) => {
    if (array1.length != array2.length) {
        throw(`Cannot add arrays of unequal length: ${array1.length}!=${array2.length}`)
    };
    let result = array1.map((num, idx) => {
        return num + array2[idx];
    })
    return result;
}

export { 
    arrayRange, 
    sumArray,
    addArrays,
    scalarMultiplyArray,
}