#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Numerical integral using Trapezoidal rule: \int x(t) dt
# Input:
#   - x [array] : integrand
#   - t [array] : domain of x

auc = function(x, t) {
  trapezoidT = diff(t)
  trapezoidX = c(x[1] + x[2], diff(cumsum(x), lag = 2))
  area = sum(trapezoidT * trapezoidX/2)
  return (area)
}


# Example
x = c(1, 2, 3)
t = 0:2
auc(x = x, t = t)