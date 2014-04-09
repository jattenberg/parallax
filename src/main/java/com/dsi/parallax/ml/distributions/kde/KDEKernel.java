/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions.kde;

// TODO: Auto-generated Javadoc
/**
 * The Enum KDEKernel.
 */
public enum KDEKernel {
    
    /** The uniform. */
    UNIFORM {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return 0.5*indicator(Math.abs(dist)<=1);
        }
        
    },
    
    /** The triangular. */
    TRIANGULAR {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return (1-Math.abs(dist))*indicator(Math.abs(dist)<=1);
        }
        
    },
    
    /** The epanechnikov. */
    EPANECHNIKOV {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return (3./4.)*(1.-dist*dist)*indicator(Math.abs(dist)<=1);
        }
        
    },
    
    /** The quartic. */
    QUARTIC {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return Math.abs(dist)>1 ? 0 : (15./16.)*Math.pow((1-dist*dist), 3.);

        }
        
    },
    
    /** wikipedia has 35/32, this gives > 1 distances. */
    TRIWEIGHT {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return Math.abs(dist)>1 ? 0 : (32./35.)*Math.pow((1-Math.pow(dist, 2)), 3);
        }
        
    }, 
    
    /** The tricube. */
    TRICUBE {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return Math.abs(dist)>1 ? 0 : (70./81)*Math.pow((1-Math.pow(Math.abs(dist), 3)), 3);
        }
        
    },
    
    /** The gaussian. */
    GAUSSIAN {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return Math.pow(2*Math.PI,-.5)*Math.exp(-0.5*Math.pow(dist, 2));
            
        }
        
    },
    
    /** The cosine. */
    COSINE {

        @Override
        public double similarity(double center, double distantPoint,
                double bandwidth) {
            double dist = dist(center, distantPoint, bandwidth);
            return Math.abs(dist)>1 ? 0 : (Math.PI/4.)*Math.cos(dist*Math.PI/2.);
        }
        
    };
    
    /**
     * Indicator.
     *
     * @param statement the statement
     * @return the double
     */
    private static double indicator(boolean statement){return statement?1:0;}
    
    /**
     * Dist.
     *
     * @param center the center
     * @param distantPoint the distant point
     * @param bandwidth the bandwidth
     * @return the double
     */
    private static double dist(double center, double distantPoint, double bandwidth){return (center-distantPoint)/bandwidth;}
    
    /**
     * Similarity.
     *
     * @param center the center
     * @param distantPoint the distant point
     * @return the double
     */
    public double similarity(double center, double distantPoint) {return similarity (center, distantPoint, 1);} 
    
    /**
     * Similarity.
     *
     * @param center the center
     * @param distantPoint the distant point
     * @param bandwidth the bandwidth
     * @return the double
     */
    public abstract double similarity(double center, double distantPoint, double bandwidth);
}
