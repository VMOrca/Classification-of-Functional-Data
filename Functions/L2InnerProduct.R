L2InnerProduct = function(x, t) {
  trapezoidT = diff(t)
  trapezoidX = c(x[1] + x[2], diff(cumsum(x), lag = 2))
  area = sum(trapezoidT * trapezoidX/2)
  return (area)
}
