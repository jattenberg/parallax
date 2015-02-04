/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection.featureselection;

import com.dsi.parallax.ml.projection.AbstractProjection;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;

import java.util.Arrays;

/**
 * Base class for all feature selection methods; Methods that attempt to find the 
 * most meaningful subset of dimensions from some problem space. Treated as a 
 * projection, where some dimensions are projected untouched, and simply "shifted up".
 * Other features are removed.
 */
public abstract class AbstractFeatureSelection extends AbstractProjection implements FeatureSelection {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -5583816285971781377L;
    
    /** The selected features; Those features thought to be most useful.  */
    protected int[] selectedFeatures;
    
    /** The input dimension of the problem space. */
    protected final int inDim;
    
    /** The output dimension; the number of features selected. */
    protected int outDim;
    
    /**
     * Instantiates a new abstract feature selection.
     *
     * @param inDim The input dimension of the problem space.
     * @param outDim The output dimension; the number of features selected.
     */
    protected AbstractFeatureSelection(int inDim, int outDim) {
    	super(inDim, outDim);
        this.inDim = inDim;
        selectedFeatures = new int[this.inDim];
        this.outDim = outDim;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.featureselection.FeatureSelection#setNumberToKeep(int)
     */
    @Override
    public void setNumberToKeep(int outDim){
        this.outDim = outDim;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.featureselection.FeatureSelection#getNumberToKeep()
     */
    @Override
    public int getNumberToKeep(){
        return outDim;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.Projection#project(com.parallax.ml.util.vector.LinearVector)
     */
    @Override
    public LinearVector project(LinearVector x) {
        LinearVector lv = LinearVectorFactory.getVector(outDim);
        for(int i = 0; i < outDim; i++) 
            lv.updateValue(i, x.getValue(selectedFeatures[i]));
        return lv;
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.projection.featureselection.FeatureSelection#getKeptFeatures()
     */
    @Override
	public int[] getKeptFeatures() {
    	return Arrays.copyOf(selectedFeatures, selectedFeatures.length);
    }
}
