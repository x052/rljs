// https://stackoverflow.com/questions/15313418/what-is-assert-in-javascript
function assert (condition, message) {
  if (!condition) {
    message = message || 'Assertion failed'
    if (typeof Error !== 'undefined') {
      throw new Error(message)
    }
    throw message
  }
}

// Random numbers utils
var returnValue = false
var vValue = 0.0
function gaussRandom () {
  if (returnValue) {
    returnValue = false
    return vValue
  }
  var u = 2 * Math.random() - 1
  var v = 2 * Math.random() - 1
  var r = u * u + v * v
  if (r === 0 || r > 1) return gaussRandom()
  var c = Math.sqrt(-2 * Math.log(r) / r)
  vValue = v * c // cache this
  returnValue = true
  return u * c
}

function randf (a, b) {
  return Math.random() * (b - a) + a
}

function randi (a, b) {
  return Math.floor(Math.random() * (b - a) + a)
}

function randn (mu, std) {
  return mu + gaussRandom() * std
}

// helper function returns array of zeros of length n
// and uses typed arrays if available
function zeros (n) {
  if (typeof (n) === 'undefined' || isNaN(n)) { return [] }
  if (typeof ArrayBuffer === 'undefined') {
      // lacking browser support
    var arr = new Array(n)
    for (var i = 0; i < n; i++) { arr[i] = 0 }
    return arr
  } else {
    return new Float64Array(n)
  }
}

var sampleWeighted = function (p) {
  var r = Math.random()
  var c = 0.0
  for (var i = 0, n = p.length; i < n; i++) {
    c += p[i]
    if (c >= r) { return i }
  }
  assert(false, 'wtf')
}

var setConst = function (arr, c) {
  for (var i = 0, n = arr.length; i < n; i++) {
    arr[i] = c
  }
}

function getopt (opt, fieldName, defaultValue) {
  if (typeof opt === 'undefined') {
    return defaultValue
  }
  return (typeof opt[fieldName] !== 'undefined') ? opt[fieldName] : defaultValue
}

module.exports = {
  assert,
  randf,
  randi,
  randn,
  zeros,
  setConst,
  getopt,
  sampleWeighted
}
