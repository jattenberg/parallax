/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.projection;

import com.dsi.parallax.ml.projection.PolynomialHashProjection;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

// TODO: Auto-generated Javadoc
/**
 * The Class RandomPolynomialHashProjectionPipe.
 */
public class RandomPolynomialHashProjectionPipe extends AbstractPipe<LinearVector, LinearVector> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -9066541110143196489L;
    
    /** The poly proj. */
    private PolynomialHashProjection polyProj;
    
    /**
     * Instantiates a new random polynomial hash projection pipe.
     *
     * @param polyProj the poly proj
     */
    public RandomPolynomialHashProjectionPipe(PolynomialHashProjection polyProj) {
    	super();
        this.polyProj = polyProj;
    }
    
    /**
     * Instantiates a new random polynomial hash projection pipe.
     *
     * @param indim the indim
     * @param outdim the outdim
     * @param degree the degree
     */
    public RandomPolynomialHashProjectionPipe(int indim, int outdim, int degree) {
        this(new PolynomialHashProjection(indim, outdim, degree));
    }

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<RandomPolynomialHashProjectionPipe>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
        LinearVector vec = context.getData();
        LinearVector out = polyProj.project(vec);
        return Context.createContext(context, out);
	}

}
