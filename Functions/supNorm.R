# Function to calculate Lp norm in Hilbert spaces, i.e. ||f||_p = (\int |f(x)|^p dx)^(1/p)
supNorm = function(x) {
  out = max(abs(x))
  return (out)
}

# Example
# x <- seq(0, pi, len = 101)
# y <- sin(x)
# SupNorm(x, y, p = 1)