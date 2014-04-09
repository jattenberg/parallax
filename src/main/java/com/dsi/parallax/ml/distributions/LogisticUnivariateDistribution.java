/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;

import com.dsi.parallax.ml.util.MLUtils;

// TODO: Auto-generated Javadoc
/**
 * The Class LogisticUnivariateDistribution.
 */
public class LogisticUnivariateDistribution extends AbstractUnivariateDistribution {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -8128552322880810647L;
    
    /** The gaussian. */
    private UnivariateGaussianDistribution gaussian; 
    
    /**
     * Instantiates a new logistic univariate distribution.
     */
    public LogisticUnivariateDistribution() {
        gaussian = new UnivariateGaussianDistribution();
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#observe(double)
     */
    @Override
    public void observe(double value) {
        gaussian.observe(value);
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#probability(double)
     */
    @Override
    public double probability(double value) {
        double m = gaussian.getMean();
        double s = gaussian.getVariance();
        
        return MLUtils.floatingPointEquals(s,0.) ? 0 :
             Math.pow(MLUtils.sech((value-m)/(2.*s)), 2.)/(4.*s);
    }

}
