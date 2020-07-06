# Function to calculate Lp norm in Hilbert spaces, i.e. ||f||_p = (\int |f(x)|^p dx)^(1/p)
LpNorm = function(x, t, p = 2) {
  xP = abs(x)^p
  trapezoidT = diff(t)
  trapezoidX = c(xP[1] + xP[2], diff(cumsum(xP), lag = 2))
  area = sum(trapezoidT * trapezoidX/2)^(1/p)
  return (area)
}

# Example
# x <- seq(0, pi, len = 101)
# y <- sin(x)
# LpNorm(x, y, p = 1)


