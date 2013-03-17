/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

// TODO: Auto-generated Javadoc
/**
 * The Class MultinomialDistribution.
 */
public class MultinomialDistribution extends AbstractDistribution {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 4497892773827113646L;

	/** The vector. */
	private LinearVector vector;

	/** The sum. */
	private double alpha = 0, sum;

	/**
	 * Instantiates a new multinomial distribution.
	 *
	 * @param size the size
	 */
	public MultinomialDistribution(int size) {
		super(size);
		vector = LinearVectorFactory.getVector(this.size);
		sum = 0;
	}

	/**
	 * Instantiates a new multinomial distribution.
	 *
	 * @param size the size
	 * @param alpha the alpha
	 */
	public MultinomialDistribution(int size, double alpha) {
		this(size);
		this.alpha = alpha;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#observe(int, double)
	 */
	@Override
	public void observe(int dimension, double value) {
		if (dimension >= size)
			throw new ArrayIndexOutOfBoundsException("attempting to add "
					+ dimension + " to array of size " + size);
		vector.updateValue(dimension, value);
		sum += value;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#probability(int)
	 */
	@Override
	public double probability(int dimension) {
		return (vector.getValue(dimension) + alpha) / (sum + size * alpha);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#probability(int, double)
	 */
	@Override
	public double probability(int dimension, double value) {
		return Math.pow(probability(dimension), value);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.distributions.Distribution#logProbability(int, double)
	 */
	@Override
	public double logProbability(int dimension, double value) {
		return logProbability(dimension) * value;
	}
}
