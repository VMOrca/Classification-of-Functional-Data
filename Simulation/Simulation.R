# Simulation: Construct sample paths of stochastic processes and use sample paths as functional data for classification 
library(hawkes)

#===============================================================================================================================
# Ornstein-Uhlenbeck process

# Function to calculate Ito integral \int_tMin^tMax f(t) dX, where X is a semimartingale
itoIntegral = function(f, dX = 'dWt', partition = 100, tMin = 0, tMax = 1, ...) {
  t = seq(tMin, tMax, length.out = partition)
  fDiscretised = f(t, ...)
  switch(dX, 
         dWt = {
           Xt = rnorm(n = partition, mean = 0, sd = sqrt(t))
         })
  dXt = diff(Xt)
  out = sum(fDiscretised[-length(t)] * dXt)
  return(out)
}

# Function to generate sample paths of Ornstein-Uhlenbeck process using Euler-Maruyama scheme
# - n: number of sample paths to be generated
# - tehta, mu, sigma: parameters in an OU process
# - x0: x(0) = starting value at t = 0
# - tMax: upper limit of time-axis
# - partition: {number of partitions + 1} in [0, tMax]
OrnsteinUhlenbeck = function(n, theta, mu, sigma, x0, tMax, partition = 100) {
  t = seq(0, tMax, length.out = partition)
  f = function(s, u) {exp(-theta * (u - s))}
  xt = matrix(rep(0, n * partition), nrow = n)
  for (i in 1:n) {
    integral = rep(0, partition)
    for (j in 2:partition) {
      integral[j] = itoIntegral(f = f, dX = 'dWt', partition = partition, tMin = 0, tMax = t[j], u = t[j])
    }
    xt[i, ] = x0 * exp(-theta * t) + mu * (1 - exp(-theta * t)) + sigma * integral
  }
  colnames(xt) = t
  return(xt)
}

# Function to plot trajectories of generated OU process
plotOrnsteinUhlenbeck = function(OUSample, xMin, xMax, tMax, partition, title = 'Ornstein-Uhlenbeck Process Trajectory') {
  t = seq(0, tMax, length.out = partition)
  plot(t, OUSample[1, ], type = 'l', ylim = c(xMin, xMax), xlab = 't', ylab = 'x(t)', 
       main = title)
  for (i in 1:n) {
    lines(t, OUSample[i, ], col = i)
  }
}

# Start simulating OU process
# 2 OU processes:
# - OUSample0 and OUSample1 has different expected value (i.e. different drift parameter mu), but all other parameters are the same
n = 200
theta = 1
mu = 10
sigma = 0.3
x0 = 5
tMax = 10
partition = 101
set.seed(1)

OUSample0 = OrnsteinUhlenbeck(n = n, theta = theta, mu = mu, sigma = sigma, x0 = x0, tMax = tMax, partition = partition)

mu = 10.2
OUSample1 = OrnsteinUhlenbeck(n = n, theta = theta, mu = mu, sigma = sigma, x0 = x0, tMax = tMax, partition = partition)

# Plot
par(mfrow = c(2, 1))
xMin = min(min(OUSample0, OUSample1))
xMax = max(max(OUSample0, OUSample1))
plotOrnsteinUhlenbeck(OUSample = OUSample0, xMin = xMin, xMax = xMax, tMax = tMax, partition = partition, 
                      title = paste('Ornstein-Uhlenbeck Process Trajectory\n (mu = ', 10, ', sigma = ', sigma, ')', sep = ''))
plotOrnsteinUhlenbeck(OUSample = OUSample1, xMin = xMin, xMax = xMax, tMax = tMax, partition = partition, 
                      title = paste('Ornstein-Uhlenbeck Process Trajectory\n (mu = ', 10.2, ', sigma = ', sigma, ')', sep = ''))

# Assemble OUSample0 and OUSample1 to a data frame
dfOUSample0 = data.frame(id = 1:n, 
                         label = 0, 
                         OUSample0)
colnames(dfOUSample0)[-(1:2)] = gsub('X', '', colnames(dfOUSample0)[-(1:2)])

dfOUSample1 = data.frame(id = (n + 1):(2 * n), 
                         label = 1, 
                         OUSample1)
colnames(dfOUSample1)[-(1:2)] = gsub('X', '', colnames(dfOUSample1)[-(1:2)])

# Final dataframe to be used for classification
dfOUDifMu = rbind(dfOUSample0, dfOUSample1)



# Start simulating OU process
# 2 OU processes:
# - OUSample2 and OUSample3 has different variance (i.e. different volatility parameter sigma), but all other parameters are the same
n = 200
theta = 1
mu = 10
sigma = 0.3
x0 = 5
tMax = 10
partition = 101
set.seed(1)

OUSample2 = OrnsteinUhlenbeck(n = n, theta = theta, mu = mu, sigma = sigma, x0 = x0, tMax = tMax, partition = partition)

sigma = 0.4
OUSample3 = OrnsteinUhlenbeck(n = n, theta = theta, mu = mu, sigma = sigma, x0 = x0, tMax = tMax, partition = partition)

# Plot
par(mfrow = c(2, 1))
xMin = min(min(OUSample2, OUSample3))
xMax = max(max(OUSample2, OUSample3))
plotOrnsteinUhlenbeck(OUSample = OUSample2, xMin = xMin, xMax = xMax, tMax = tMax, partition = partition, 
                      title = paste('Ornstein-Uhlenbeck Process Trajectory\n (mu = ', mu, ', sigma = ', 0.3, ')', sep = ''))
plotOrnsteinUhlenbeck(OUSample = OUSample3, xMin = xMin, xMax = xMax, tMax = tMax, partition = partition, 
                      title = paste('Ornstein-Uhlenbeck Process Trajectory\n (mu = ', mu, ', sigma = ', 0.4, ')', sep = ''))


# Assemble OUSample0 and OUSample1 to a data frame
dfOUSample2 = data.frame(id = 1:n, 
                         label = 0, 
                         OUSample2)
colnames(dfOUSample2)[-(1:2)] = gsub('X', '', colnames(dfOUSample2)[-(1:2)])

dfOUSample3 = data.frame(id = (n + 1):(2 * n), 
                         label = 1, 
                         OUSample3)
colnames(dfOUSample3)[-(1:2)] = gsub('X', '', colnames(dfOUSample3)[-(1:2)])

# Final dataframe to be used for classification
dfOUDifSigma = rbind(dfOUSample2, dfOUSample3)








#===============================================================================================================================
# Hawkes Process
# Function to calculate exponential decay
mu = function(t, alpha, beta) {
  out = alpha * exp(-beta * t)
  return(out)
}

# Function to calculate sample path for intensity process (not Hawkes process generated by the intensity process!)
hawkesIntensity = function(t, tJump, ...) {
  hawkesSamplePath = rep(NA, length(t))
  for (i in 1:length(t)){
    if (length(which(tJump < t[i])) >= 1) {
      tJumpLoop = tJump[which(tJump <  t[i])]
      hawkesSamplePath[i] = lambda0 + sum(mu(t = t[i] - tJumpLoop, alpha = alpha, beta = beta))
    } else {
      hawkesSamplePath[i] = lambda0
    }
  }
  return(hawkesSamplePath)
}


# # Generate 1 realisation of Hawkes process and plot it (Just an example)
# lambda0 = 2
# alpha = 2
# beta = 25
# tMax = 360
# hawkesJump = simulateHawkes(lambda0, alpha, beta, tMax)[[1]]
# 
# # par(mfrow = c(2, 1))
# t = 0:tMax
# intensitySamplePath = hawkesIntensity(t = t, tJump = hawkesJump, alpha = alpha, beta = beta)
# plot(t, intensitySamplePath, type = 'l', xlab = 't', ylab = 'Intensity',
#      main = 'Intensity Process of a Hawkes Process')
# hawkesSamplePath = 1:length(hawkesJump)
# plot(hawkesJump, hawkesSamplePath, type = 'l', xlab = 't', ylab = 'Count',
#      main = 'Sample Path of a Hawkes Process')



# Generate two groups of Hawkes process, which will be used for classification
lambda0 = 2
alpha = 2
beta = 20
tMax = 360
n = 200
t = 0:tMax
set.seed(1)

# Group 0
intensitySamplePath0 = matrix(rep(NA, n * length(t)), nrow = n)
for (i in 1:n) {
  hawkesJump = simulateHawkes(lambda0, alpha, beta, tMax)[[1]]
  intensitySamplePath0[i, ] = hawkesIntensity(t = t, tJump = hawkesJump, alpha = alpha, beta = beta)
}
dfHawkesIntensity0 = data.frame(id = 1:n, 
                                label = 0, 
                                intensitySamplePath0)
colnames(dfHawkesIntensity0)[-(1:2)] = as.character(t)

# Group 1
beta = 25
intensitySamplePath1 = matrix(rep(NA, n * length(t)), nrow = n)
for (i in 1:n) {
  hawkesJump = simulateHawkes(lambda0, alpha, beta, tMax)[[1]]
  intensitySamplePath1[i, ] = hawkesIntensity(t = t, tJump = hawkesJump, alpha = alpha, beta = beta)
}
dfHawkesIntensity1 = data.frame(id = (n + 1):(2 * n), 
                                label = 1, 
                                intensitySamplePath1)
colnames(dfHawkesIntensity1)[-(1:2)] = as.character(t)

# Assemble to final data frame, which will be used for classification
dfHawkes = rbind(dfHawkesIntensity0, dfHawkesIntensity1)




# Function to plot trajectories of generated OU process
plotHawkes = function(t, HawkesSample, xMin, xMax) {
  n = dim(HawkesSample)[1]
  plot(t, HawkesSample[1, ], type = 'l', ylim = c(xMin, xMax), xlab = 't', ylab = 'Intensity', 
       main = 'Intensity Process of a Hawkes Process')
  for (i in 1:n) {
    lines(t, HawkesSample[i, ], col = i)
  }
}

# Plot Hawkes process
par(mfrow = c(2, 1))
hawkes0 = select(dfHawkesIntensity0, -id, -label)
hawkes1 = select(dfHawkesIntensity1, -id, -label)
xMin = min(min(hawkes0, hawkes1))
xMax = max(max(hawkes0, hawkes1))
plotHawkes(t = t, HawkesSample = hawkes0, xMin = xMin, xMax = xMax)
plotHawkes(t = t, HawkesSample = hawkes1, xMin = xMin, xMax = xMax)
