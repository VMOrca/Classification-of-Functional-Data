#########################################################################################################################################
#
#                                                       Author: Min Sun
#
#########################################################################################################################################

# Function to calculate supremum norm in Hilbert spaces, i.e. ||x||_\infty = sup(|x(t)|)
# Input:
#   - x [array] : value of x at each t

supNorm = function(x) {
  out = max(abs(x))
  return (out)
}

# Example
# x <- seq(0, pi, len = 101)
# supNorm(x)