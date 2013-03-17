/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

import java.io.Serializable;

// TODO: Auto-generated Javadoc
/**
 * The Interface UnivariateDistribution.
 */
public interface UnivariateDistribution extends Serializable {
    
    /**
     * Observe.
     *
     * @param value the value
     */
    public void observe(double value);
    
    /**
     * Probability.
     *
     * @param value the value
     * @return the double
     */
    public double probability(double value);
    
    /**
     * Log probability.
     *
     * @param value the value
     * @return the double
     */
    public double logProbability(double value);
}
