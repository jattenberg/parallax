/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.projection;

import java.lang.reflect.Type;
import java.util.List;

import com.dsi.parallax.ml.projection.RandomPCAProjection;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractAccumulatingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Collections2;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * projects linear vectors onto a random subspace, then onto an eigen space
 * optionally compiles the eigen vectors on accumulated vectors
 * TODO: make prettier.
 *
 * @author jattenberg
 */
public class RandomPCAProjectionPipe extends AbstractAccumulatingPipe<LinearVector, LinearVector> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -1686691310617294337L;
    
    /** The rpca. */
    private RandomPCAProjection rpca;


    /**
     * Instantiates a new random pca projection pipe.
     *
     * @param pca the pca
     */
    public RandomPCAProjectionPipe(RandomPCAProjection pca) {
        this(pca, -1);
    }
    
    /**
     * Instantiates a new random pca projection pipe.
     *
     * @param pca the pca
     * @param toConsider the to consider
     */
    public RandomPCAProjectionPipe(RandomPCAProjection pca, int toConsider) {
        super(toConsider);
    	this.rpca = pca;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.pipeline.AbstractAccumulatingPipe#isTrained()
     */
    @Override
    public boolean isTrained() {
    	return rpca.isBuilt();
    }

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<RandomPCAProjectionPipe>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		return Context.createContext(context, rpca.project(context.getData()));
	}
	
	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#batchProcess(java.util.List)
	 */
	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		rpca.build(Collections2.transform(infoList, uncontextifier));
	}

}
