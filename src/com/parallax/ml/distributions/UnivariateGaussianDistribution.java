/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

// TODO: Auto-generated Javadoc
/**
 * The Class UnivariateGaussianDistribution.
 */
public class UnivariateGaussianDistribution extends AbstractUnivariateDistribution {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -3883288413689138238L;
    
    /** The var_part. */
    private double mean, observations, var_part;
    
    /**
     * Instantiates a new univariate gaussian distribution.
     */
    public UnivariateGaussianDistribution(){ mean=0; observations=0; var_part=0; }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#observe(double)
     */
    @Override
    public void observe(double value) {
        observations++;
        double delta = value - mean;
        mean += delta/observations;
        var_part += delta*(value-mean);
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#probability(double)
     */
    @Override
    public double probability(double datum) {
        if(observations==0)
            return 0;
        double variance = var_part/(observations-1);
        return  gaussian(datum, mean, variance);
        
    }

    /**
     * Gaussian.
     *
     * @param datum the datum
     * @param mean the mean
     * @param variance the variance
     * @return the double
     */
    public static double gaussian(double datum, double mean, double variance) {
        return Math.pow(2*Math.PI*variance,-.5)*Math.exp(-Math.pow(datum-mean, 2.)/(2*variance));
    }

    /**
     * Gets the mean.
     *
     * @return the mean
     */
    public double getMean() {
        return mean;
    }
    
    /**
     * Gets the variance.
     *
     * @return the variance
     */
    public double getVariance() {
        return var_part/(observations>1?observations-1:observations);
    }
    
    
}
