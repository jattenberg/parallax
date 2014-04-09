/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.base.Function;

/**
 * ContextAddingFunction create Context
 *
 * @param <I>
 * @author Josh Attenberg
 */
public class ContextAddingFunction<I> implements Function<I, Context<I>>{

	@Override
	public Context<I> apply(I in) {
		return Context.createContext(in);
	}
}
