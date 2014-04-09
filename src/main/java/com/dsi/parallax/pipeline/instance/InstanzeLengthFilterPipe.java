/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import java.lang.reflect.Type;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.pipeline.AbstractFilteringPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * The Class InstanzeLengthFilterPipe.
 *
 * @param <I> the generic type
 */
public class InstanzeLengthFilterPipe<I extends Instance<?>> extends AbstractFilteringPipe<I> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = 6743546404481494213L;
    
    /** The length. */
    private int length = 0;
    
    /**
     * Instantiates a new instanze length filter pipe.
     */
    public InstanzeLengthFilterPipe() {
        super();
    }
    
    /**
     * Instantiates a new instanze length filter pipe.
     *
     * @param length the length
     */
    public InstanzeLengthFilterPipe(int length) {
    	this();
        this.length = length;
    }

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<InstanzeLengthFilterPipe<I>>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractFilteringPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected boolean operate(Context<I> context) {
		Instance<?> payload = context.getData();
		return Math.round(payload.L0Norm()) > length;
	}
}
