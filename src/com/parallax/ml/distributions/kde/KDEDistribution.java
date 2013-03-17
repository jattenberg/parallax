/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions.kde;

import com.parallax.ml.distributions.AbstractDistribution;

// TODO: Auto-generated Javadoc
/**
 * The Class KDEDistribution.
 */
public class KDEDistribution extends AbstractDistribution {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 4189498106787152199L;
    
    /** The kdes. */
    private UnivariateKernelDensityEstimator[] kdes;
    
    /** The kernel. */
    private KDEKernel kernel = KDEKernel.GAUSSIAN;
    
    /** The bandwidth. */
    private int bandwidth = 50; 
    
    /** The distance damping. */
    private double distanceDamping = 1;
    
    /**
     * Instantiates a new kDE distribution.
     *
     * @param size the size
     */
    public KDEDistribution(int size) {
        super(size);
        kdes = new UnivariateKernelDensityEstimator[this.size];
    }
    
    /**
     * Instantiates a new kDE distribution.
     *
     * @param size the size
     * @param kde the kde
     */
    public KDEDistribution(int size, KDEKernel kde) {
        this(size);
        this.kernel = kde;
    }
    
    /**
     * Instantiates a new kDE distribution.
     *
     * @param size the size
     * @param bandwidth the bandwidth
     * @param distanceDamping the distance damping
     * @param kde the kde
     */
    public KDEDistribution(int size, int bandwidth, double distanceDamping, KDEKernel kde) {
        this(size, kde);
        this.bandwidth = bandwidth;
        this.distanceDamping = distanceDamping;
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.Distribution#observe(int, double)
     */
    @Override
    public void observe(int dimension, double value) {
        if(null == kdes[dimension])
            kdes[dimension] = new UnivariateKernelDensityEstimator(bandwidth, distanceDamping, kernel);
        kdes[dimension].observe(value);
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.Distribution#probability(int)
     */
    @Override
    public double probability(int dimension) {
        return probability(dimension, 1);
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.Distribution#probability(int, double)
     */
    @Override
    public double probability(int dimension, double value) {
        if(null == kdes[dimension])
            return 0;
        else
            return kdes[dimension].probability(value);
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.Distribution#logProbability(int, double)
     */
    @Override
    public double logProbability(int dimension, double value) {
        if(null == kdes[dimension])
            return 0;
        else
            return Math.log(kdes[dimension].probability(value));
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.AbstractDistribution#logProbability(int)
     */
    @Override
    public double logProbability(int dimension) {
        return Math.log(probability(dimension,1));
    }
}
