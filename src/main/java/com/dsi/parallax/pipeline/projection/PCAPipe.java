/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.projection;

import com.dsi.parallax.ml.projection.PrincipalComponentsAnalysis;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractAccumulatingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Collections2;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

// TODO: Auto-generated Javadoc
/**
 * projects linear vectors onto a eigen space
 * optionally compiles the eigen vectors on accumulated vectors.
 *
 * @author jattenberg
 */
public class PCAPipe extends AbstractAccumulatingPipe<LinearVector, LinearVector> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -1686691310617294337L;
    
    /** The pca. */
    private PrincipalComponentsAnalysis pca;
    
    /**
     * Instantiates a new pCA pipe.
     *
     * @param pca the pca
     */
    public PCAPipe(PrincipalComponentsAnalysis pca) {
        this(pca, -1);
    }

    /**
     * Instantiates a new pCA pipe.
     *
     * @param pca the pca
     * @param toConsider the to consider
     */
    public PCAPipe(PrincipalComponentsAnalysis pca, int toConsider) {
        super(toConsider);
        this.pca = pca;
    }
    
    /* (non-Javadoc)
     * @see com.parallax.pipeline.AbstractAccumulatingPipe#isTrained()
     */
    @Override
    public boolean isTrained() {
        return pca.isBuilt();
    }
    
	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<PCAPipe>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		return Context.createContext(context, pca.project(context.getData()));
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#batchProcess(java.util.List)
	 */
	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		pca.build(Collections2.transform(infoList, uncontextifier));
	}
}
