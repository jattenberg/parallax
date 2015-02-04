/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import com.dsi.parallax.ml.target.Target;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

// TODO: Auto-generated Javadoc
/**
 * The Class MeanScorer.
 *
 * @param <T> the generic type
 */
public class MeanScorer<T extends Target> extends AbstractObjectiveScorer<T> {

	/** The stats. */
	private DescriptiveStatistics stats; 
	
	/**
	 * Instantiates a new mean scorer.
	 *
	 * @param objective the objective
	 */
	public MeanScorer(Objective<T> objective) {
		super(objective);
		reset();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#getScore()
	 */
	@Override
	public double getScore() {
		return stats.getMean();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#evaluate(double)
	 */
	@Override
	public  double evaluate(
			double partialScore) {
		stats.addValue(partialScore);
		return partialScore;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#reset()
	 */
	@Override
	public void reset() {
		stats = new DescriptiveStatistics();
	}

}
