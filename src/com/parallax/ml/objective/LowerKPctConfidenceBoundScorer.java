/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import static com.google.common.base.Preconditions.checkArgument;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.TDistribution;
import org.apache.commons.math.distribution.TDistributionImpl;
import org.apache.commons.math.stat.descriptive.SummaryStatistics;

import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * returns the lower bound of the 95% confidence interval as the score.
 * 
 * @param <T>
 *            the generic type
 * @author jattenberg
 */
public class LowerKPctConfidenceBoundScorer<T extends Target> extends
		AbstractObjectiveScorer<T> {

	/** The stats. */
	private SummaryStatistics stats;

	/** The Constant SIGNIFICANCE. */
	private static final double DEFAULT_SIGNIFICANCE = .95;

	private final double significance;

	/**
	 * Instantiates a new k pct confidence bound scorer.
	 * 
	 * @param objective
	 *            the objective
	 */
	public LowerKPctConfidenceBoundScorer(Objective<T> objective) {
		this(DEFAULT_SIGNIFICANCE, objective);
	}

	/**
	 * Instantiates a new k pct confidence bound scorer.
	 * 
	 * @param objective
	 *            the objective
	 */
	public LowerKPctConfidenceBoundScorer(double significance,
			Objective<T> objective) {
		super(objective);
		reset();
		checkArgument(significance > 0. && significance <= 1.,
				"significance value must be in (0, 1], given: %s", significance);
		this.significance = significance;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.objective.ObjectiveScorer#getScore()
	 */
	@Override
	public double getScore() {
		TDistribution tDist = new TDistributionImpl(stats.getN() - 1);
		double a = 0;
		try {
			a = tDist.inverseCumulativeProbability(significance);
		} catch (MathException e) {
			throw new RuntimeException(e);
		}
		double width = a * stats.getStandardDeviation()
				/ Math.sqrt(stats.getN());
		double mean = stats.getMean();
		return mean - width;
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
		stats = new SummaryStatistics();
	}

}
