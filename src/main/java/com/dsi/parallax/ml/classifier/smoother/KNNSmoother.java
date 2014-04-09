package com.dsi.parallax.ml.classifier.smoother;

import com.dsi.parallax.ml.util.KDTree.Entry;

/**
 * smooths by finding the k nearest neighbors to the input raw prediction,
 * and supplies the laplace-smoothed mean label amongst these neighbors
 */
public class KNNSmoother extends AbstractKNNSmoother<KNNSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -7053159685861146532L;

	public KNNSmoother() {
		super();
	}

	/**
	 * Instantiates a new kNN smoother.
	 * 
	 * @param k
	 *            the number of neighboring score points to consider
	 */
	public KNNSmoother(int k) {
		super(k);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		double sum = 0, count = 0;

		for (Entry<Double> entry : findNeighbors(prediction)) {
			count++;
			sum += entry.value;
		}

		return (1. + sum) / (2. + count);
	}

	@Override
	public KNNSmoother getSmoother() {
		return this;
	}

}
