/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import com.parallax.ml.target.Target;

/**
 * Compiles multiple {@link Objective} values (for instance, objectives computed
 * during each fold of cross validation, and compiles into the
 * {@link <a href="http://en.wikipedia.org/wiki/Geometric_mean">Geometric Mean</a>}
 * 
 * @param <T>
 *            Type of target being considered.
 */
public class GeometricMeanScorer<T extends Target> extends
		AbstractObjectiveScorer<T> {

	/** The stats. */
	private DescriptiveStatistics stats;

	/**
	 * Instantiates a new geometric mean scorer.
	 * 
	 * @param objective
	 *            the objective function being scored.
	 */
	public GeometricMeanScorer(Objective<T> objective) {
		super(objective);
		reset();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.objective.ObjectiveScorer#getScore()
	 */
	@Override
	public double getScore() {
		return stats.getGeometricMean();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.objective.ObjectiveScorer#evaluate(double)
	 */
	@Override
	public double evaluate(double partialScore) {
		stats.addValue(partialScore);
		return partialScore;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.objective.ObjectiveScorer#reset()
	 */
	@Override
	public void reset() {
		stats = new DescriptiveStatistics();
	}

}
