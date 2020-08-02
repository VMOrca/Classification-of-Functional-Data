#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to calculate Lp norm in Hilbert spaces, i.e. ||x||_p = (\int |x(t)|^p dt)^(1/p)
# Input:
#   - x [array] : integrand
#   - t [array] : domain of x
#   - p [int] : positive integer choosing which Lp norm to be used

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


