/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.vector.LinearVector;
import com.google.common.collect.Maps;

import java.util.Map;

public class RoundingTruncation extends AbstractGradientTruncation {

	private static final long serialVersionUID = -163666191365079129L;

	RoundingTruncation(int period, double threshold) {
		this.period = period;
		setTheta(threshold);
	}

	@Override
	public LinearVector truncateParameters(LinearVector vector) {
		if (++epoch % period != 0) {
			return vector;
		}

		Map<Integer, Double> w = Maps.newHashMap();
		for (int i : vector) {
			double input = vector.getValue(i);
			double output = Math.abs(input) < getTheta() ? 0 : input;
			w.put(i, output);
		}

		for (int i : w.keySet()) {
			vector.resetValue(i, w.get(i));
		}
		return vector;
	}

	@Override
	public TruncationType getType() {
		return TruncationType.ROUNDING;
	}

}
