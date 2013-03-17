/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * The Class MinScorer.
 *
 * @param <T> the generic type
 */
public class MinScorer<T extends Target> extends AbstractObjectiveScorer<T> {

	/** The stats. */
	private double score;
	
	/**
	 * Instantiates a new min scorer.
	 *
	 * @param objective the objective
	 */
	public MinScorer(Objective<T> objective) {
		super(objective);
		reset();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#getScore()
	 */
	@Override
	public double getScore() {
		return score == Double.MAX_VALUE ? 0 : score;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#evaluate(double)
	 */
	@Override
	public double evaluate(
			double partialScore) {
		score = Math.min(score, partialScore);	
		return partialScore;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#reset()
	 */
	@Override
	public void reset() {
		score = Double.MAX_VALUE;
	}

}
