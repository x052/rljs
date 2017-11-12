const { assert, zeros } = require('./utils')

class Matrix {
  constructor (rows, columns) {
    this.rows = rows
    this.columns = columns

    this.w = zeros(rows * columns)
    this.dw = zeros(rows * columns)
  }

  get (row, col) {
    // slow but careful accessor function
    // we want row-major order
    var ix = (this.columns * row) + col
    assert(ix >= 0 && ix < this.w.length)
    return this.w[ix]
  }
  set (row, col, v) {
    // slow but careful accessor function
    var ix = (this.columns * row) + col
    assert(ix >= 0 && ix < this.w.length)
    this.w[ix] = v
  }

  setFrom (arr) {
    for (var i = 0, n = arr.length; i < n; i++) {
      this.w[i] = arr[i]
    }
  }
  setColumn (m, i) {
    for (var q = 0, n = m.w.length; q < n; q++) {
      this.w[(this.columns * q) + i] = m.w[q]
    }
  }
  toJSON () {
    var json = {}
    json['rows'] = this.rows
    json['columns'] = this.columns
    json['w'] = this.w
    return json
  }
  fromJSON (json) {
    this.rows = json.rows
    this.columns = json.columns
    this.w = zeros(this.rows * this.columns)
    this.dw = zeros(this.rows * this.columns)
    for (var i = 0, n = this.rows * this.columns; i < n; i++) {
      this.w[i] = json.w[i] // copy over weights
    }
  }
}

function copyMatrix (b) {
  var a = new Matrix(b.rows, b.columns)
  a.setFrom(b.w)
  return a
}

module.exports = {
  Matrix,
  copyMatrix
}
