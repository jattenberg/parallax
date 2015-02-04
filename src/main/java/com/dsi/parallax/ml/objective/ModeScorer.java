/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import com.dsi.parallax.ml.target.Target;
import com.google.common.collect.Maps;

import java.util.Map;

// TODO: Auto-generated Javadoc
/**
 * The Class ModeScorer.
 *
 * @param <T> the generic type
 */
public class ModeScorer<T extends Target> extends AbstractObjectiveScorer<T> {

	/** The vals. */
	private Map<Double, Double> vals;
	
	/** The max value. */
	private double maxValue = Double.MIN_VALUE;
	
	/** The max. */
	private double max;
	
	/**
	 * Instantiates a new mode scorer.
	 *
	 * @param objective the objective
	 */
	public ModeScorer(Objective<T> objective) {
		super(objective);
		reset();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#getScore()
	 */
	@Override
	public double getScore() {
		return max;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#evaluate(double)
	 */
	@Override
	public double evaluate(
			double partialScore) {
		vals.put(partialScore, (vals.containsKey(partialScore) ? vals.get(partialScore) + 1 : 1));		
		if(vals.get(partialScore) > maxValue) {
			maxValue = vals.get(partialScore);
			max = partialScore;
		}
		return partialScore;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#reset()
	 */
	@Override
	public void reset() {
		vals = Maps.newHashMap();
		maxValue = Double.MIN_VALUE;
	}
	


}
