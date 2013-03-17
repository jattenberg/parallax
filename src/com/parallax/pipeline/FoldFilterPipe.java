/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
/**
 * passes one in folds instances,
 * indexed by fold < folds
 * 
 * @author jattenberg
 *
 * @param <O>
 */
public class FoldFilterPipe<O>  extends AbstractFilteringPipe<O> {

    private static final long serialVersionUID = 8438497807962209870L;
    private final int fold, folds;
    private int ct;

    /**
     * Class constructor specifying number of fold and number of folds to create
     * @param fold number of fold
     * @param folds number of folds
     */
    public FoldFilterPipe(int fold, int folds) {
        super();
    	if(fold < 0 || folds <= 0 || fold >= folds)
            throw new IllegalArgumentException("folds must be positive (given: " + folds + "), fold must be non-neg, (given: " + fold + ") and folds must be > fold");
        this.fold = fold;
        this.folds = folds;
        ct = 0;
    }

    /**
     * The method returns the class's Type "FoldFilterPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<FoldFilterPipe<O>>(){}.getType();
	}

	@Override
	protected boolean operate(Context<O> context) {
		return ct++%folds == fold;
	}
}
