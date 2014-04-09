/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

public interface PipeFilter<T> {
    public boolean filterContext(Context<T> context);
}
