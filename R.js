const { randf, randn } = require('./utils')
const { Matrix, copyMatrix } = require('./Matrix')

function copyNet (net) {
  // nets are (k,v) pairs with k = string key, v = Mat()
  var newNet = {}
  for (var p in net) {
    if (net.hasOwnProperty(p)) {
      newNet[p] = copyMatrix(net[p])
    }
  }
  return newNet
}

function updateMatrix (m, alpha) {
  // updates in place
  for (var i = 0, n = m.n * m.d; i < n; i++) {
    if (m.dw[i] !== 0) {
      m.w[i] += -alpha * m.dw[i]
      m.dw[i] = 0
    }
  }
}

function updateNet (net, alpha) {
  for (var p in net) {
    if (net.hasOwnProperty(p)) {
      updateMatrix(net[p], alpha)
    }
  }
}

function netToJSON (net) {
  var j = {}
  for (var p in net) {
    if (net.hasOwnProperty(p)) {
      j[p] = net[p].toJSON()
    }
  }
  return j
}

function netFromJSON (j) {
  var net = {}
  for (var p in j) {
    if (j.hasOwnProperty(p)) {
      net[p] = new Matrix(1, 1) // not proud of this
      net[p].fromJSON(j[p])
    }
  }
  return net
}

function netZeroGrads (net) {
  for (var p in net) {
    if (net.hasOwnProperty(p)) {
      var mat = net[p]
      gradFillConst(mat, 0)
    }
  }
}

function netFlattenGrads (net) {
  var n = 0
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      let mat = net[p]; n += mat.dw.length
    }
  }
  var g = new Matrix(n, 1)
  var ix = 0
  for (let p in net) {
    if (net.hasOwnProperty(p)) {
      var mat = net[p]
      for (var i = 0, m = mat.dw.length; i < m; i++) {
        g.w[ix] = mat.dw[i]
        ix++
      }
    }
  }
  return g
}

// return Matrix but filled with random numbers from gaussian
function RandMatrix (n, d, mu, std) {
  var m = new Matrix(n, d)
  fillRandn(m, mu, std)
  // fillRand(m,-std,std); // kind of :P
  return m
}

// Mat utils
// fill matrix with random gaussian numbers
function fillRandn (m, mu, std) {
  for (var i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randn(mu, std)
  }
}

function fillRand (m, lo, hi) {
  for (var i = 0, n = m.w.length; i < n; i++) {
    m.w[i] = randf(lo, hi)
  }
}

function gradFillConst (m, c) {
  for (var i = 0, n = m.dw.length; i < n; i++) {
    m.dw[i] = c
  }
}

function softmax (m) {
  var out = new Matrix(m.n, m.d) // probability volume
  var maxval = -999999
  for (let i = 0, n = m.w.length; i < n; i++) {
    if (m.w[i] > maxval) maxval = m.w[i]
  }

  var s = 0.0
  for (let i = 0, n = m.w.length; i < n; i++) {
    out.w[i] = Math.exp(m.w[i] - maxval)
    s += out.w[i]
  }
  for (let i = 0, n = m.w.length; i < n; i++) {
    out.w[i] /= s
  }

    // no backward pass here needed
    // since we will use the computed probabilities outside
    // to set gradients directly on m
  return out
}

var initLSTM = function (inputSize, hiddenSizes, outputSize) {
  // hidden size should be a list

  var model = {}
  for (var d = 0; d < hiddenSizes.length; d++) { // loop over depths
    var prevSize = d === 0 ? inputSize : hiddenSizes[d - 1]
    var hiddenSize = hiddenSizes[d]

    // gates parameters
    model['Wix' + d] = new RandMatrix(hiddenSize, prevSize, 0, 0.08)
    model['Wih' + d] = new RandMatrix(hiddenSize, hiddenSize, 0, 0.08)
    model['bi' + d] = new Matrix(hiddenSize, 1)
    model['Wfx' + d] = new RandMatrix(hiddenSize, prevSize, 0, 0.08)
    model['Wfh' + d] = new RandMatrix(hiddenSize, hiddenSize, 0, 0.08)
    model['bf' + d] = new Matrix(hiddenSize, 1)
    model['Wox' + d] = new RandMatrix(hiddenSize, prevSize, 0, 0.08)
    model['Woh' + d] = new RandMatrix(hiddenSize, hiddenSize, 0, 0.08)
    model['bo' + d] = new Matrix(hiddenSize, 1)
    // cell write params
    model['Wcx' + d] = new RandMatrix(hiddenSize, prevSize, 0, 0.08)
    model['Wch' + d] = new RandMatrix(hiddenSize, hiddenSize, 0, 0.08)
    model['bc' + d] = new Matrix(hiddenSize, 1)
  }
  // decoder params
  model['Whd'] = new RandMatrix(outputSize, hiddenSize, 0, 0.08)
  model['bd'] = new Matrix(outputSize, 1)
  return model
}

function forwardLSTM (G, model, hiddenSizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  var hiddenPrevs = []
  var cellPrevs = []
  if (prev == null || typeof prev.h === 'undefined') {
    for (let d = 0; d < hiddenSizes.length; d++) {
      hiddenPrevs.push(new Matrix(hiddenSizes[d], 1))
      cellPrevs.push(new Matrix(hiddenSizes[d], 1))
    }
  } else {
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  var hidden = []
  var cell = []
  for (let d = 0; d < hiddenSizes.length; d++) {
    var inputVector = d === 0 ? x : hidden[d - 1]
    var hiddenPrev = hiddenPrevs[d]
    var cellPrev = cellPrevs[d]

    // input gate
    var h0 = G.mul(model['Wix' + d], inputVector)
    var h1 = G.mul(model['Wih' + d], hiddenPrev)
    var inputGate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + d]))

    // forget gate
    var h2 = G.mul(model['Wfx' + d], inputVector)
    var h3 = G.mul(model['Wfh' + d], hiddenPrev)
    var forgetGate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + d]))

    // output gate
    var h4 = G.mul(model['Wox' + d], inputVector)
    var h5 = G.mul(model['Woh' + d], hiddenPrev)
    var outputGate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + d]))

    // write operation on cells
    var h6 = G.mul(model['Wcx' + d], inputVector)
    var h7 = G.mul(model['Wch' + d], hiddenPrev)
    var cellWrite = G.tanh(G.add(G.add(h6, h7), model['bc' + d]))

    // compute new cell activation
    var retainCell = G.eltmul(forgetGate, cellPrev) // what do we keep from cell
    var writeCell = G.eltmul(inputGate, cellWrite) // what do we write to cell
    var cellD = G.add(retainCell, writeCell) // new cell contents

    // compute hidden state as gated, saturated cell activations
    var hiddenD = G.eltmul(outputGate, G.tanh(cellD))

    hidden.push(hiddenD)
    cell.push(cellD)
  }

  // one decoder to outputs at end
  var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]), model['bd'])

  // return cell memory, hidden representation and output
  return {'h': hidden, 'c': cell, 'o': output}
}

var maxi = function (w) {
  // argmax of array w
  var maxv = w[0]
  var maxix = 0
  for (var i = 1, n = w.length; i < n; i++) {
    var v = w[i]
    if (v > maxv) {
      maxix = i
      maxv = v
    }
  }
  return maxix
}

var samplei = function (w) {
  // sample argmax from w, assuming w are
  // probabilities that sum to one
  var r = randf(0, 1)
  var x = 0.0
  var i = 0
  while (true) {
    x += w[i]
    if (x > r) {
      return i
    }
    i++
  }
  return w.length - 1 // pretty sure we should never get here?
}

module.exports = {
  // utils
  softmax,
  samplei,
  maxi,
  // classes
  RandMatrix,
  forwardLSTM,
  initLSTM,
  // more utils
  updateMatrix,
  updateNet,
  copyMatrix,
  copyNet,
  netToJSON,
  netFromJSON,
  netZeroGrads,
  netFlattenGrads
}
