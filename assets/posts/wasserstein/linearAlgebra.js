function Matrix(rows){
    let nrows = rows.length;
    let ncols = rows[0].length;
    for (let i = 0; i < nrows; i++){
        if (rows[i].length != ncols) {throw "all rows must have the same number of columns"}
    }
    this.matrix = rows.map(row => row.slice()); //deep copy
    this.nrows = nrows;
    this.ncols = ncols;
}

Matrix.prototype.toString = function(){return this.matrix};

Matrix.prototype.get = function(i, j) {
    if (i >= this.nrows || j >= this.ncols) {throw `index (${i}, ${j}) out of bounds for ${this.nrows}x${this.ncols} matrix`}
    return this.matrix[i][j];
}

// constructors
const zeros = (nrows, ncols) => {
    return  new Matrix([...Array(nrows)].map(e => Array(ncols).fill(0)) );
}

const matrixFromDiagonal = (diagonal) => {
    let n = diagonal.length;
    let m = zeros(n, n)
    for (let i=0; i < n; i++){
        m.matrix[i][i] = diagonal[i];
    }
    return m;
}

// Element wise
Matrix.prototype.elementWiseOperation = function(op, s) {
    let result = zeros(this.nrows, this.ncols);
    for (let i = 0; i < this.nrows; i++){
        for (let j = 0; j < this.ncols; j ++){
            let total = op(this.matrix[i][j], s);
            result.matrix[i][j] = total;
        }
    }
    return result;
}

Matrix.prototype.addScalar = function(s){return this.elementWiseOperation((x, y) => x + y, s)};
Matrix.prototype.subtractScalar = function(s){return this.elementWiseOperation((x, y) => x - y, s)};
Matrix.prototype.multiplyScalar = function(s){return this.elementWiseOperation((x, y) => x * y, s)};

// self operations
Matrix.prototype.transpose = function(){
    let result = zeros(this.ncols, this.nrows);
    for (let i = 0; i < this.nrows; i++){
        for (let j = 0; j < this.ncols; j ++){
            result.matrix[j][i] = this.matrix[i][j];
        }
    }
    return result;
}

Matrix.prototype.trace = function(){
    let result = 0.0;
    let nrows = this.matrix.length;
    for (let i = 0; i < nrows; i++){
            result += this.matrix[i][i];
    }
    return result;
}

Matrix.prototype.inv = function(){
    if (this.nrows != 2 || this.ncols != 2) {throw "Only implemented for 2x2 matrices"}
    let a = this.matrix[0][0];
    let b = this.matrix[0][1];
    let c = this.matrix[1][0];
    let d = this.matrix[1][1];
    let determinant = a*d - b*c
    let result = new Matrix([[d/determinant, -b/determinant], [-c/determinant, a/determinant]]);
    return result;
}

// matrix-matrix operations
const matrixMultiply = (A, B) => {
    if (A.ncols != B.nrows){
        throw `${A.nrows}x${A.ncols} * ${B.nrows}x${B.ncols}, matrix inner dimensions do not match`;
    }
    let result = zeros(A.nrows, B.ncols);
    let total;
    for (let i = 0; i < A.nrows; i++){
        for (let j = 0; j < B.ncols; j ++){
            total = 0.0;
            for (let k = 0; k < A.ncols; k ++){
                total += A.get(i, k) * B.get(k, j);
            }
            result.matrix[i][j] = total;
        }
    }
    return result;
}

const elementWiseOperation = (op, A, B, symbol='+') => {
    const message = `${A.nrows}x${A.ncols} ${symbol} ${B.nrows}x${A.ncols}`;
    if (A.nrows != B.nrows) throw `${message}, row dimensions do not match`
    if (A.ncols != B.ncols) throw `${message}, column dimensions do not match`

    let result = zeros(A.nrows, A.ncols);
    let total = 0.0;
    for (let i = 0; i < A.nrows; i++){
        for (let j = 0; j < A.ncols; j ++){
            total = op(A.get(i, j), B.get(i, j));
            result.matrix[i][j] = total;
        }
    }
    return result;
}

const matrixAdd = (A, B) => elementWiseOperation((x, y) => x + y, A, B, '+');
const matrixSubtract = (A, B) => elementWiseOperation((x, y) => x - y, A, B, '-');
const matrixMultiplyElementWise = (A, B) => elementWiseOperation((x, y) => x * y, A, B, '*');
const matrixDivideElementWise = (A, B) => elementWiseOperation((x, y) => x / y, A, B, '/');

// complex self operations

Matrix.prototype.eigen = function(atol=1e-6) {
    if (this.nrows != 2 || this.ncols != 2) {throw "Only implemented for 2x2 matrices"}
    let a = this.matrix[0][0];
    let b = this.matrix[0][1];
    let c = this.matrix[1][0];
    let d = this.matrix[1][1];

    if ((Math.abs(b) < atol) && (Math.abs(c) < atol))
    {
        let lambdas = a >= d ? [d, a] : [a, d]
        let vectors = a >= d ? [[0.0, 1.0], [1.0, 0.0]] : [[1.0, 0.0], [0.0, 1.0]]
        return {eigenvalues: lambdas, eigenvectors: vectors}
    } 

    let discriminant = (a - d) * (a - d) + 4 * b * c;
    if (discriminant < 0) throw ("complex eigenvalues are not supported");
    let delta = 0.5 * Math.sqrt(discriminant);
    let lambdas = [(a + d) / 2 - delta, (a + d) / 2 + delta];

    let vector0, vector1;
    if (Math.abs(c) < atol){
        let v0 =  lambdas[0] - a;
        let mag0 = Math.sqrt(b * b + v0 * v0);
        vector0 = [b / mag0, v0 / mag0]
        let v1 =  lambdas[1] - a;
        let mag1 = Math.sqrt(b * b + v1 * v1);
        vector1 = [b / mag1, v1 / mag1]
    }
    else {
        let v0 =  lambdas[0] - d;
        let mag0 = Math.sqrt(v0 * v0 + c * c);
        vector0 = [v0 / mag0, c / mag0];
        let v1 =  lambdas[1] - d;
        let mag1 = Math.sqrt(v1 * v1 + c * c);
        vector1 = [v1 / mag1, c / mag1]  
    }
    
    return {eigenvalues: lambdas, eigenvectors: [vector0, vector1]}
}

Matrix.prototype.sqrt = function(atol){
    let eigens = this.eigen(atol);
    eigens.eigenvalues = eigens.eigenvalues.map((val) => {
        if (val < 0){
           if(Math.abs(val) > atol) throw "complex eigenvalues are not supported";
           return val = 0;
        }
        else {
            return val
        }
    })
    let W = new Matrix(eigens.eigenvectors).transpose();
    let Winv = W.inv();
    let sqrtEigens = eigens.eigenvalues.map(x => Math.sqrt(x));
    let D = matrixFromDiagonal(sqrtEigens);
    let result = matrixMultiply(W, matrixMultiply(D, Winv))
    return result;
}

export {
    Matrix, 
    matrixMultiply, 
    matrixAdd, 
    matrixSubtract, 
    matrixMultiplyElementWise, 
    matrixDivideElementWise
}