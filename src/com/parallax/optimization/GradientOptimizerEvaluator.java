/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization;


/*
 *  Callback interface that allows optimizer clients to perform some operation after every iteration.
 */
public interface GradientOptimizerEvaluator {
    /**
     * Performs some operation at the end of each iteration of a maximizer.
     *
     * @param maxable Function that's being optimized.
     * @param iter    Number of just-finished iteration.
     * @return true if optimization should continue.
     */
    boolean evaluate (GradientOptimizable maxable, int iter);
}
