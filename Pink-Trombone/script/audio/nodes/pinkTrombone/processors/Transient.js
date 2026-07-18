class Transient {
    constructor(position, seconds) {
        this.position = position;

        this.startTime = seconds;
        this.timeAlive = 0;
        this.lifetime = 0.2;

        this.strength = 0.3;
        this.exponent = 200;
    }

    get amplitude() {
        // Exponential decay with a 1/exponent half-life (5ms). The original
        // Pink Trombone computes strength * 2^(-exponent * timeAlive); writing
        // pow(-2, timeAlive * exponent) instead returns NaN for non-integer
        // exponents (negative base).
        return this.strength * Math.pow(2, -this.exponent * this.timeAlive);
    }

    get isAlive() {
        return this.timeAlive < this.lifetime;
    }

    update(seconds) {
        this.timeAlive = seconds - this.startTime;
    }
}

export default Transient;