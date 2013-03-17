/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import java.util.Collection;

import com.google.common.collect.Lists;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.vector.LinearVector;

// TODO: Auto-generated Javadoc
/**
 * The Class RandomPCAProjection.
 */
public class RandomPCAProjection extends AbstractConstructedProjection
{

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 2168966068533270718L;
    
    /** The Constant SALT. */
    private static final int SALT = 1234567;
    
    /** The hashproj. */
    private HashProjection hashproj;
    
    /** The pca. */
    private PrincipalComponentsAnalysis pca;
    
    /**
     * Instantiates a new random pca projection.
     *
     * @param inputDim the input dim
     * @param intermediateDim the intermediate dim
     * @param outDim the out dim
     */
    public RandomPCAProjection(int inputDim, int intermediateDim, int outDim)
	{
    	super(inputDim, outDim);
        hashproj = new HashProjection(inputDim, intermediateDim, SALT);
        pca = new PrincipalComponentsAnalysis(intermediateDim, outDim);
	}
    
    /**
     * Instantiates a new random pca projection.
     *
     * @param inputDim the input dim
     * @param intermediateDim the intermediate dim
     * @param outDim the out dim
     * @param X the x
     */
    public RandomPCAProjection(int inputDim, int intermediateDim, int outDim, Collection<LinearVector> X)
    {
        this(inputDim,intermediateDim, outDim);
        build(X);
    }

    /**
     * Instantiates a new random pca projection.
     *
     * @param inputDim the input dim
     * @param intermediateDim the intermediate dim
     * @param outDim the out dim
     * @param X the x
     */
    public RandomPCAProjection(int inputDim, int intermediateDim, int outDim, Instances<?> X)
    {
        this(inputDim,intermediateDim, outDim, X.getFeatureVectors());
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.ConstructedProjection#build(java.util.Collection)
     */
    @Override
    public void build(Collection<LinearVector> X) {
        Collection<LinearVector> Y = Lists.newLinkedList();
        for(LinearVector x : X)
            Y.add(hashproj.project(x));
        pca.build(Y);
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.ConstructedProjection#isBuilt()
     */
    @Override
    public boolean isBuilt() {
        return pca.isBuilt();
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.Projection#project(com.parallax.ml.util.vector.LinearVector)
     */
    @Override
    public LinearVector project(LinearVector x) {
        LinearVector intermediate = hashproj.project(x);
        return pca.project(intermediate);
    }

}
