/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.vector.LinearVector;
import com.google.common.collect.Maps;

import java.util.Map;

public class PegasosTruncation extends AbstractGradientTruncation {

	private static final long serialVersionUID = -4780712211976054363L;

	PegasosTruncation(int period, double alpha) {
		this.period = period;
		this.alpha = alpha;
	}

	@Override
	public LinearVector truncateParameters(LinearVector vector) {
		if (++epoch % period != 0 || vector.L2Norm() == 0 || alpha <= 0)
			return vector;

		Map<Integer, Double> w = Maps.newHashMap(); // for avoiding concurrent
													// modifications
		double numerator = 1d / Math.sqrt(alpha);
		double denominator = vector.L2Norm();

		double ratio = numerator / denominator;


		if (ratio > 1) {
			return vector;
		}
			
		for (int i : vector) {
			w.put(i, vector.getValue(i) * ratio);
		}

		for (int i : w.keySet()) {
			vector.resetValue(i, w.get(i));
		}
		return vector;
	}

	@Override
	public TruncationType getType() {
		return TruncationType.PEGASOS;
	}

}
