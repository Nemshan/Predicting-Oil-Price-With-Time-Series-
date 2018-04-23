# Authors: 
#   Michael Gryncewicz
#   Nemshan Alharthi
# Revision Date:
#   April 22, 2018
# Contact email:
#   mrl98@wildcats.unh.edu
#   naa1014@wildcats.unh.edu
# 
# The goal of this project was to use various time series methods in order to
# make predictions on a dataset of our choice. The modeling techniques to be used 
# were ARIMA models, Exponential Smoothing models, and Facebooks Prophet model.
# 
# The data used for our time series project came from the link below. 
# This data consists of monthly West Texas Intermediate oil prices from Cushing,
# Oklahoma. The data extends from January 1986 up to February 2018. 
#   https://fred.stlouisfed.org/series/MCOILWTICO
#
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
# import libraries
library(fpp2)
library(astsa)
library(tseries)
library(forecast)
library(dplyr)
library(prophet)

# data source:
# https://fred.stlouisfed.org/series/MCOILWTICO

# load in data and make into time series
setwd('/Users/MikeGryncewicz/Documents/College/05 Grad School/03 Spring/Data 902/time_series/final_project/')
oil <- read.csv('MCOILWTICO.csv')

head(oil) # look at the data
oil_ts <- ts(oil$MCOILWTICO, start = c(1986, 1), frequency = 12)
plot(oil_ts) # visualizing the time series

train <- window(oil_ts, c(1986, 1), c(2017, 8)) # training data - all data except last six months
test <- window(oil_ts, c(2017, 9), c(2018, 2) ) # test data - just the last six months

plot(train) # visualizing train and test data
lines(test, col = 'red') 

################################################################################################################
# ARIMA Model

# Step 1: Visualize the time series
plot(oil_ts) # obvious trend and increasing variance

# Step 2: Check for stationarity
plot(diff(log(train))) # using a diff and a log to stationarize the data

adf.test(train)
adf.test(diff(log(train))) # passes the Dickey Fuller test

acf2(train) # acf tails off - pacf kinda tails sorta

auto.arima(train) # suggests: p=2, d=1, q=2, P=0, D=0, Q=2, S=12

# first model --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
fit <- sarima(log(train), p=2, d=1, q=2, P=0, D=0, Q=2, S=12)
print(cbind(fit$AIC, fit$BIC)) # look at AIC and BIC of model
fit$ttable 

sarima_oil <- as.ts(sarima.for(train, n.ahead = 6, p=2, d=1, q=2, P=0, D=0, Q=2, S=12))
lines(test, col = 'blue') # compare predicted and actual
accuracy(sarima_oil$pred, test) # look at accuracy of model


# second model --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
fit2 <- sarima(log(train), p=2, d=1, q=0, P=0, D=0, Q=0, S=12)
print(cbind(fit2$AIC, fit2$BIC)) # look at AIC and BIC of model
fit2$ttable 

sarima_oil2 <- as.ts(sarima.for(train, n.ahead = 6, p=2, d=1, q=0, P=0, D=0, Q=1, S=12))
lines(test, col = 'blue') # compare predicted and actual
accuracy(sarima_oil2$pred, test) # look at accuracy of model

# third model --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
fit3 <- sarima(log(train), p=2, d=1, q=2, P=1, D=0, Q=2, S=12)
print(cbind(fit3$AIC, fit3$BIC)) # look at AIC and BIC of model
fit3$ttable 

sarima_oil3 <- as.ts(sarima.for(train, n.ahead = 6, p=2, d=1, q=2, P=1, D=0, Q=2, S=12))
lines(test, col = 'blue') # compare predicted and actual
accuracy(sarima_oil3$pred, test) # look at accuracy of model

# forth model --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
fit4 <- sarima(log(train), p=2, d=1, q=1, P=1, D=0, Q=1, S=12)
print(cbind(fit4$AIC, fit4$BIC)) # look at AIC and BIC of model
fit4$ttable 

sarima_oil4 <- as.ts(sarima.for(train, n.ahead = 6, p=2, d=1, q=1, P=1, D=0, Q=1, S=12))
lines(test, col = 'blue') # compare predicted and actual
accuracy(sarima_oil4$pred, test) # look at accuracy of model

# ARIMA Summary:
# first model was best - slightly less accurate than third model but less complex
# AR only model was not as good as first model even though suggested by ACF and PACF plots


################################################################################################################
# Exponetial Smoothing Model

# not trying simple exponetial smoothing due to trend in data

# holt model
fit_holt <- holt(train, h = 12)
summary(fit_holt)
checkresiduals(fit_holt) # checking for autocorrelation in residuals
autoplot(fit_holt) + autolayer(fitted(fit_holt)) # plotting model fit
accuracy(fit_holt, test) # checking model accuracy

for_fit <- forecast(fit_holt, h = 6) # forecast test values
plot(for_fit) # plot the predicted data
lines(test, col = 'red')

# holt model damped
fit_holt_d <- holt(train, h = 12, damped = TRUE)
summary(fit_holt_d)
checkresiduals(fit_holt_d)
autoplot(fit_holt_d) + autolayer(fitted(fit_holt_d))
accuracy(fit_holt_d, test)

for_fit2 <- forecast(fit_holt_d, h = 6)
plot(for_fit2)
lines(test, col = 'red')

# holt winters model - dont expect this model to work super well - no clear seasonality in the data
fit_hw <- hw(train, seasonal = "multiplicative")
summary(fit_hw)
checkresiduals(fit_hw)
autoplot(fit_hw) + autolayer(fitted(fit_hw))
accuracy(fit_hw, test)

for_fit3 <- forecast(fit_hw, h = 6)
plot(for_fit3)
lines(test, col = 'red')

# Exponential Smoothing Summary:
# holt model seems to predict the best
# makes sense as there is a trend but no seasonality

################################################################################################################
# Phophet Model
head(oil) 
colnames(oil)[1] <- 'ds' # renaming columns for prophet modeling
colnames(oil)[2] <- 'y'

oil_train = oil[1:380,] # re-subsetting data - prophet requires dataframe instead of ts
oil_test = oil[381:386,]

prophet_fit <- prophet(oil_train) 
future <- make_future_dataframe(prophet_fit, periods = 6, freq = 'month') # make df for forecasted predictions
forecast <- predict(prophet_fit, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(prophet_fit, forecast)
prophet_plot_components(prophet_fit, forecast)
accuracy(forecast[c('yhat')][380:386,], test) # checking model accuracy

# Prophet model summary:
# fits better than exponential smoothing models but not as well as final ARIMA model


