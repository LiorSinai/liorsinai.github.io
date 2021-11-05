"use strict";

class Matrix{
    constructor(array){
        let nrows = array.length;
        let ncols = array[0].length;
        for (var i = 0; i < nrows; i++){
            if (array[i].length != ncols) throw "all rows must have the same number of columns"
        }
        this.matrix = array.map(row => row.slice()); //deep copy
    }

    static zeros = (nrows, ncols) => {
        return  new Matrix( [...Array(nrows)].map(e => Array(ncols).fill(0)) );
    }

    index = (...idxs) => {
        let v = this.matrix;
        idxs.forEach(i => v = v[i])
        return v
    }

    multiply = (M) => {
        let nrows1 = this.matrix.length;
        let ncols1 = this.matrix[0].length;
        let nrows2 = M.matrix.length;
        let ncols2 = M.matrix[0].length;
        if (ncols1 != nrows2) throw nrows1 + "x" + ncols1 + " * " + nrows2 + "x" + ncols1 + ", matrix inner dimensions do not match"
        let result = Matrix.zeros(nrows1, ncols2);

        let total;
        for (var i = 0; i < nrows1; i++){
            for (var j = 0; j < ncols2; j ++){
                total = 0;
                for (var k = 0; k < ncols1; k ++){
                    total += this.matrix[i][k] * M.matrix[k][j];
                }
                result.matrix[i][j] = total;
            }
        }
        return result;
    }

    tranpose = () => {
        let nrows = this.matrix.length;
        let ncols = this.matrix[0].length;
        let result = Matrix.zeros(ncols, nrows);
        
        for (var i = 0; i < nrows; i++){
            for (var j = 0; j < ncols; j ++){
                result.matrix[j][i] = this.matrix[i][j];
            }
        }
        return result;
    }
}

export const getTaitBryanZYX = (yaw, pitch, roll) => {
    // yaw (Z), pitch (Y), roll (X) convention
    let c, s;
    c = Math.cos(yaw);
    s = Math.sin(yaw);
    let Ryaw = new Matrix([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ]);

    c = Math.cos(pitch);
    s = Math.sin(pitch);
    let Rpitch = new Matrix([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ]);

    c = Math.cos(roll);
    s = Math.sin(roll);
    let Rroll = new Matrix([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ]);

    let R = Rroll.multiply(Rpitch).multiply(Ryaw);
    return R;
}

export const rotateWithR = (v, R) => {
    let vM = new Matrix([ [v[0]] , [v[1]], [v[2]] ]);
    let vR = R.multiply(vM);
    return [vR.matrix[0][0], vR.matrix[1][0], vR.matrix[2][0]]; 
}

export{ Matrix }