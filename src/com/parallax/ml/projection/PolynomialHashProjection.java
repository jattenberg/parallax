/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import java.util.Set;

import com.google.common.collect.Sets;
import com.parallax.ml.util.MLUtils;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

// TODO: Auto-generated Javadoc
/**
 * TODO: avoid tricks with bigInts, just do several hash-and-checks.
 *
 * @author jattenberg
 */
public class PolynomialHashProjection extends AbstractProjection {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -4487318140264755425L;
    
    /** The degrees. */
    private final int degrees;
    
    /** The salt. */
    private final int salt;
    
    /** The hashcode. */
    private int hashcode = 0;
    
    /**
     * Instantiates a new polynomial hash projection.
     *
     * @param in the in
     * @param out the out
     */
    public PolynomialHashProjection(int in, int out) {
        this(in, out, 2);
    }

    /**
     * Instantiates a new polynomial hash projection.
     *
     * @param in the in
     * @param out the out
     * @param degree the degree
     */
    public PolynomialHashProjection(int in, int out, int degree) {
        this(in, out, degree, 12345);
    }

    /**
     * Instantiates a new polynomial hash projection.
     *
     * @param in the in
     * @param out the out
     * @param degree the degree
     * @param salt the salt
     */
    public PolynomialHashProjection(int in, int out, int degree, int salt) {
    	super(in, out);
    	this.degrees = degree;
        this.salt = salt;
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.projection.Projection#project(com.parallax.ml.util.vector.LinearVector)
     */
    @Override
    public LinearVector project(LinearVector x) {
        LinearVector out = LinearVectorFactory.getVector(outDim);
        
        
        for (int degree = 0; degree < degrees; degree++) {
            for (int x_i : x) {
                Set<Integer> indices = getIndexesInRange(x_i + " " + degree + " " + salt);
                
                for(int x_new : indices) {
                    double psi = MLUtils.boxMullerHash(x_i + " " + x_new + " " + degree + " "
                            + salt + " psi");
                    double y_new = psi * Math.pow(x.getValue(x_i), degree);
                    out.updateValue(x_new, y_new);
                }
            }
        }
        return out;
    }
    
    /**
     * for a given input, pick a "random" set of output indices in the range.
     *
     * @param input string
     * @return set bits
     */
    private Set<Integer> getIndexesInRange(String input) {
        Set<Integer> out = Sets.newHashSet();
        for(int i = 0; i < outDim; i++) {
            int binhash = Math.abs((input+" "+i).hashCode());
            if(binhash%2==1)
                out.add(i);
        }
        return out;
    }

    /* (non-Javadoc)
     * @see java.lang.Object#hashCode()
     */
    @Override
    public int hashCode() {
        int result = hashcode;
        // lazy implementation
        if (result == 0) {
            result = 17;
            result = 31 * result + inDim;
            result = 31 * result + outDim;
            result = 31 * result + salt;
            result = 31 * result + degrees;
            hashcode = result;
        }
        return result;
    }

    /* (non-Javadoc)
     * @see java.lang.Object#equals(java.lang.Object)
     */
    @Override
    public boolean equals(Object o) {
        if (!(o instanceof PolynomialHashProjection))
            return false;
        PolynomialHashProjection p = (PolynomialHashProjection) o;
        return (p.salt == salt && p.inDim == inDim && p.outDim == outDim && p.degrees == degrees);
    }
}
