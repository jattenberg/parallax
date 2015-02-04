/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.vector.LinearVector;
import com.google.common.collect.Maps;

import java.util.Map;

public class TruncatedGradient extends AbstractGradientTruncation {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7609740993220832690L;

	TruncatedGradient(int period, double alpha, double theta) {
		this.period = period;
		this.alpha = alpha;
		setTheta(theta);
	}

	@Override
	public LinearVector truncateParameters(LinearVector vector) {
		if (++epoch % period != 0)
			return vector;
		Map<Integer, Double> w = Maps.newHashMap();
		for (int i : vector) {

			double d = vector.getValue(i);

			if (d >= 0 && d <= getTheta())
				w.put(i, (Math.max(0, d - alpha)));
			else if (d < 0 && d >= -getTheta())
				w.put(i, (Math.min(0, d + alpha)));
		}
		for (int i : w.keySet()) {
			vector.resetValue(i, w.get(i));
		}
		return vector;
	}

	@Override
	public TruncationType getType() {
		return TruncationType.TRUNCATING;
	}

}
