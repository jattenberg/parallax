/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.base.Function;
/**
 * A class that implements the Function interface (from guava), whose only job is 
 * to get the data associated with the current context, i.e., to uncontextify the 
 * context 
 * 
 * @author Josh Attenberg
 *
 * @param <I>
 * 			the type whose context is to be uncontextified 
 */
public class UncontextifyingFunction<I> implements Function<Context<I>, I>{

	@Override
	public I apply(Context<I> context) {
		return context.getData();
	}

}
