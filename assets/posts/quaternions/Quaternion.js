/*
See:
- https://github.com/infusion/Quaternion.js/blob/master/quaternion.js
- https://github.com/mrdoob/three.js/blob/dev/src/math/Quaternion.js

*/
"use strict";

class Quaternion{
    constructor(s, v1, v2, v3){
        this.s = s;
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
    }

    add = (q2) => { 
        return new Quaternion(
            this.s + q2.s,
            this.v1 + q2.v1,
            this.v2 + q2.v2,
            this.v3 + q2.v3,
            );
    }

    subtract = (q2) => {
        return new Quaternion(
            this.s - q2.s,
            this.v1 - q2.v1,
            this.v2 - q2.v2,
            this.v3 - q2.v3,
            );
    }

    conjugate = () => {
        return new Quaternion(this.s, -this.v1, -this.v2, -this.v3);
    }

    magnitude2 = () => (this.s * this.s + this.v1 * this.v1 + this.v2 * this.v2 + this.v3 * this.v3);
    magnitude = () => Math.sqrt(this.magnitude2());

    innerProduct = (q2) => (this.s * q2.s + this.v1 * q2.v1 + this.v2 * q2.v2 + this.v3 * q2.v3);

    multiply = (q2) => {
        if (!isNaN(q2)){
            return new Quaternion(this.s * q2, this.v1 * q2, this.v2 * q2, this.v3 * q2);
        }

        return new Quaternion(
            this.s * q2.s - this.v1 * q2.v1 - this.v2 * q2.v2 - this.v3 * q2.v3,
            this.s * q2.v1 + this.v1 * q2.s + this.v2 * q2.v3 - this.v3 * q2.v2,
            this.s * q2.v2 - this.v1 * q2.v3 + this.v2 * q2.s + this.v3 * q2.v1,
            this.s * q2.v3 + this.v1 * q2.v2 - this.v2 * q2.v1 + this.v3 * q2.s,
        );
    }

    divide = (q2) => {
        if (!isNaN(q2)){
            return this.multiply(1/q2);
        } 
        else return this.multiply(q2.invert());
    }

    invert = () => {
        let mag2 = this.magnitude2();
        return new Quaternion(this.s/mag2, -this.v1/mag2, -this.v2/mag2, -this.v3/mag2)
    }

    normalise = () => {
        let mag = this.magnitude();
        this.s /= mag;
        this.v1 /= mag;
        this.v2 /= mag;
        this.v3 /= mag;
    }

    pm = (x) => x >= 0 ? "+" : unescape('%u2212')

    toString = (precision) =>{
        if (precision == undefined ) {precision = 4};
        let str = ""
            + this.pm(this.s) + Math.abs(this.s).toFixed(precision) + " "
            + this.pm(this.v1) + " " + Math.abs(this.v1).toFixed(precision) + "i "
            + this.pm(this.v2) + " " + Math.abs(this.v2).toFixed(precision) + "j "
            + this.pm(this.v3) + " " + Math.abs(this.v3).toFixed(precision) + "k"
        return str
    }

    static rotateVector = (v, q) => {
        let qr = q.multiply(new Quaternion(0, v[0], v[1], v[2])).multiply(q.invert());
        return [qr.v1, qr.v2, qr.v3]
    }

    static rotateAxisAngle = (v, axis, angle) => {
        let n_norm = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        let n = [axis[0]/n_norm, axis[1]/n_norm, axis[2]/n_norm]
        let sinHalfTheta = Math.sin(angle/2);
        let q = new Quaternion(
            Math.cos(angle/2),
            sinHalfTheta * n[0],
            sinHalfTheta * n[1],
            sinHalfTheta * n[2],
        );
        return Quaternion.rotateVector(v, q)
    }

    static rotateMatrix = (m, q) => {
        let ncols  = m[0].length;
        let out = [...Array(3)].map(e => Array(ncols).fill(0))
        let v, vr;
        for (var j=0; j < ncols; j++){
            v = [m[0][j], m[1][j], m[2][j]];
            vr = Quaternion.rotateVector(v, q);
            out[0][j] = vr[0];
            out[1][j] = vr[1];
            out[2][j] = vr[2];
          }
        return out;
    }

    static lerp = (q1, q2, t) => {
        t = Math.min(1, Math.max(t, 0));
        let qt = q1.multiply(1 - t).add(q2.multiply(t));
        qt.normalise();
        return qt;
    }

    static slerp = (q1, q2, t) => {
        t = Math.min(1, Math.max(t, 0)); 
        if (t === 0) return new Quaternion(q1.s, q1.v1, q1.v2, q1.v3);
        if (t === 1) return new Quaternion(q2.s, q2.v1, q2.v2, q2.v3);

        let cosOmega = q1.innerProduct(q2);
        let sinOmega = Math.sqrt(1 - cosOmega);
        if (sinOmega < Number.EPSILON) return new Quaternion(q1.s, q1.v1, q1.v2, q1.v3);

        let omega = Math.atan2(sinOmega, cosOmega);
        let ratio1 = Math.sin((1 - t) * omega) / sinOmega;
        let ratio2 = Math.sin( t * omega) / sinOmega;
        let qt = q1.multiply(ratio1).add(q2.multiply(ratio2));
        return qt;
    }
}

Quaternion.prototype.isQuaternion = true;

export{ Quaternion }