/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.regularization;

import java.util.Map;

import com.google.common.collect.Maps;
import com.parallax.ml.vector.LinearVector;

public class ModifiedDuchiTruncation extends AbstractGradientTruncation {
	private static final long serialVersionUID = 7609740993220832690L;

	ModifiedDuchiTruncation(int period, double alpha) {
		this.period = period;
		this.alpha = alpha;
	}

	@Override
	public LinearVector truncateParameters(LinearVector vector) {
		if (++epoch % period != 0)
			return vector;
		Map<Integer, Double> w = Maps.newHashMap();
		double c = Math.max(0, alpha - vector.L0Norm());
		if (c == 0) {
			return vector;
		}
		double d = vector.L0Norm();

		double tau = c / d;

		for (int i : vector) {
			double v = vector.getValue(i);
			double diff = 0;
			if (v > 0) {
				if (tau > 0) {
					diff = Math.min(v, tau);
				} else {
					diff = v;
				}
			} else if (v < 0) {
				if (tau < 0) {
					diff = Math.max(v, tau);
				} else {
					diff = v;
				}
			}

			w.put(i, Math.signum(v) * Math.abs(v - diff));
		}

		for (int i : w.keySet()) {
			vector.resetValue(i, w.get(i));
		}
		return vector;
	}

	@Override
	public TruncationType getType() {
		return TruncationType.MODDUCHI;
	}

}
