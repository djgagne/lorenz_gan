# Research Tasks
* Write forecast model script
* What set of conditioning variables are needed to produce better representations?
    * Adding X variables in time
    * Adding X variables in space
    * Value of main X variable + residuals in time and space
* What machine learning model produces the best estimates of the full pdf, especially the mean and variability?
* How should random numbers be initialized and updated to preserve the continuity of the 
unresolved state?
* What GAN configuration produces the greatest diversity of samples?
* Three main experiments
    * Weather forecast problem
    * Constant climate problem
    * Climate change problem

* Forecast metrics
    * RMSE of ensemble mean vs. ensemble spread
    * Reliability of different ensemble forecasts
    * RPSS
    * Extreme scores
    * Sharpness
    * Hellinger distance (pdf of observed state to forecast state)
* Climate Metrics
    * Quantile-quantile plot
    * Autocorrelation function in time for given X
    * Spatial autocorrelation averaged over all X locations
    * PDFs of X values and comparison with truth PDF
    * Hellinger distance on climatological PDF of forecast vs truth