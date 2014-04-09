/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.projection;

import java.lang.reflect.Type;

import com.dsi.parallax.ml.projection.HashProjection;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * The Class RandomHashProjectionPipe.
 */
public class RandomHashProjectionPipe extends AbstractPipe<LinearVector, LinearVector> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 2444390673555671034L;
    
    /** The projector. */
    private HashProjection projector;
    
    /** The Constant SALT. */
    private static final int SALT = 12345;
    
    /**
     * Instantiates a new random hash projection pipe.
     *
     * @param indim the indim
     * @param outdim the outdim
     */
    public RandomHashProjectionPipe(int indim, int outdim) {
    	super();
        projector = new HashProjection(indim, outdim, SALT);
    }

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<RandomHashProjectionPipe>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
        LinearVector payload = context.getData();
        LinearVector output = projector.project(payload);
        return Context.createContext(context, output);
	}
}
