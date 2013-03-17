/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

/**
 * Base class for univariate probability distributions. Defines a single method, 
 * {@link #logProbability(double)}
 */
public abstract class AbstractUnivariateDistribution implements UnivariateDistribution {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 2741720420838507686L;

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#logProbability(double)
     */
    @Override
    public double logProbability(double value) {
        return Math.log(probability(value));
    }

}
