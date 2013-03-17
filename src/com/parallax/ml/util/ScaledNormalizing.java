/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

public enum ScaledNormalizing {
	/**
	 * perform no scaling on input
	 */
	UNSCALED {
		@Override
		public LinearVector scaledNormalizing(LinearVector input) {
			return input;
		}
	},
	/**
	 * scale the vector by its L0 norm, by dividing each element by L0 norm
	 */
	L0 {
		@Override
		public LinearVector scaledNormalizing(LinearVector input) {
			double norm = input.L0Norm();
			return scale(input, norm);
		}
	},
	/**
	 * scale the vector by its L1 norm, by dividing each element by L1 norm
	 */
	L1 {
		@Override
		public LinearVector scaledNormalizing(LinearVector input) {
			double norm = input.L1Norm();
			return scale(input, norm);
		}
	},
	/** 
	 * scale the vector by its L2 norm, by dividing each element by L2 norm
	 */
	L2 {
		@Override
		public LinearVector scaledNormalizing(LinearVector input) {
			double norm = input.L2Norm();
			return scale(input, norm);
		}
	},
	/**
	 * scale the vector by its L-infinity norm, by dividing each element
	 * by L-infinity norm
	 */
	LINF {
		@Override
		public LinearVector scaledNormalizing(LinearVector input) {
			double norm = input.LInfinityNorm();
			return scale(input, norm);
		}
	};

	public abstract LinearVector scaledNormalizing(LinearVector input);

	private static LinearVector scale(LinearVector vec, double scale) {
		scale = MLUtils.floatingPointEquals(0, scale) ? 1 : scale;
		LinearVector out = LinearVectorFactory.getVector(vec.size());
		for (int x : vec) {
			out.resetValue(x, vec.getValue(x) / scale);
		}
		return out;
	}
}
