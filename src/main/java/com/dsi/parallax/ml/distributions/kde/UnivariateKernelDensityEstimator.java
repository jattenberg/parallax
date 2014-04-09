/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions.kde;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.TreeSet;

import com.dsi.parallax.ml.distributions.AbstractUnivariateDistribution;
import com.dsi.parallax.ml.util.MLUtils;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.common.collect.Sets;

// TODO: Auto-generated Javadoc
/**
 * The Class UnivariateKernelDensityEstimator.
 */
public class UnivariateKernelDensityEstimator extends
        AbstractUnivariateDistribution {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 6127735502737118741L;
    
    /** The points. */
    private TreeSet<Double> points;
    
    /** The bandwidth. */
    private int bandwidth = 50; 
    
    /** The distance damping. */
    private double distanceDamping = 1;
    
    /** The kernel. */
    private KDEKernel kernel = KDEKernel.GAUSSIAN;

    /**
     * Instantiates a new univariate kernel density estimator.
     */
    public UnivariateKernelDensityEstimator() {
        points = Sets.newTreeSet();
    }

    /**
     * Instantiates a new univariate kernel density estimator.
     *
     * @param kernel the kernel
     */
    public UnivariateKernelDensityEstimator(KDEKernel kernel) {
        this();
        this.kernel = kernel;
    }
    
    /**
     * Instantiates a new univariate kernel density estimator.
     *
     * @param bandwidth the bandwidth
     */
    public UnivariateKernelDensityEstimator(int bandwidth) {
        this();
        this.bandwidth = bandwidth;
        
    }

    /**
     * Instantiates a new univariate kernel density estimator.
     *
     * @param bandwidth the bandwidth
     * @param kernel the kernel
     */
    public UnivariateKernelDensityEstimator(int bandwidth, KDEKernel kernel) {
        this(bandwidth);
        this.kernel = kernel;
        
    }
    
    /**
     * Instantiates a new univariate kernel density estimator.
     *
     * @param bandwidth the bandwidth
     * @param distanceDamping the distance damping
     */
    public UnivariateKernelDensityEstimator(int bandwidth, double distanceDamping) {
        this(bandwidth);
        this.distanceDamping = distanceDamping;
    }

    /**
     * Instantiates a new univariate kernel density estimator.
     *
     * @param bandwidth the bandwidth
     * @param distanceDamping the distance damping
     * @param kernel the kernel
     */
    public UnivariateKernelDensityEstimator(int bandwidth, double distanceDamping, KDEKernel kernel) {
        this(bandwidth, distanceDamping);
        this.kernel = kernel;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#observe(double)
     */
    @Override
    public void observe(double value) {
        points.add(value);
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.distributions.UnivariateDistribution#probability(double)
     */
    @Override
    public double probability(double value) {
        NeighborIterator it = new NeighborIterator(value);
        double ct = 0;
        double tot = 0;

        while (it.hasNext() && ct++ < bandwidth) {
            double pt = it.next();
            tot += kernel.similarity(pt, value, distanceDamping);
        }

        return tot / (ct*distanceDamping);
    }

    /**
     * The Class NeighborIterator.
     */
    private class NeighborIterator implements Iterator<Double> {

        /** The query point. */
        private double queryPoint;
        
        /** The below iterator. */
        private PeekingIterator<Double> aboveIterator, belowIterator;

        /**
         * Instantiates a new neighbor iterator.
         *
         * @param queryPoint the query point
         */
        NeighborIterator(double queryPoint) {
            this.queryPoint = queryPoint;
            belowIterator = Iterators.peekingIterator(points.headSet(
                    queryPoint, false).descendingIterator());
            aboveIterator = Iterators.peekingIterator(points.tailSet(
                    queryPoint, false).iterator());
        }

        /* (non-Javadoc)
         * @see java.util.Iterator#hasNext()
         */
        @Override
        public boolean hasNext() {
            return aboveIterator.hasNext() || belowIterator.hasNext();
        }

        /* (non-Javadoc)
         * @see java.util.Iterator#next()
         */
        @Override
        public Double next() {
            if (!aboveIterator.hasNext()) {
                if (!belowIterator.hasNext()) {
                    throw new NoSuchElementException("no more elements!");
                } else {
                    return belowIterator.next();
                }
            } else if (!belowIterator.hasNext()) {
                return aboveIterator.next();
            } else {

                double above = aboveIterator.peek();
                double below = belowIterator.peek();

                double aboveSim = kernel.similarity(above, queryPoint, distanceDamping);
                double belowSim = kernel.similarity(below, queryPoint, distanceDamping);

                if (aboveSim > belowSim)
                    return aboveIterator.next();
                else if (belowSim > aboveSim)
                    return belowIterator.next();
                else {
                    if (MLUtils.GENERATOR.nextDouble() > 0.5)
                        return aboveIterator.next();
                    else
                        return belowIterator.next();
                }
            }
        }

        /* (non-Javadoc)
         * @see java.util.Iterator#remove()
         */
        @Override
        public void remove() {
            throw new UnsupportedOperationException("Remove not supported");
        }

    }

}
