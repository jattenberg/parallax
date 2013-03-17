package com.parallax.ml.classifier.smoother;

import java.util.List;

import com.google.common.collect.Lists;
import com.parallax.ml.util.KDTree.Entry;
import com.parallax.ml.util.pair.PrimitivePair;
import static com.google.common.base.Preconditions.checkArgument;

;

/**
 * The Class LocalLogisticRegressionSmoother. Combines KNN and Logistic
 * Regression, building a LR model on the K closest points to the input
 * prediction to be smoothd, and then applies this logistic regression to
 * the query point
 */
public class LocalLogisticRegressionSmoother extends
		AbstractKNNSmoother<LocalLogisticRegressionSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5719263192470426862L;

	private int passes = 10;

	/**
	 * Instantiates a new local logistic regression smoother.
	 * 
	 * @param k
	 *            the k used for KNN
	 */
	public LocalLogisticRegressionSmoother(int k) {
		super(k);
	}

	/**
	 * Instantiates a new local logistic regression smoother.
	 */
	public LocalLogisticRegressionSmoother() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.KNNSmoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		List<Entry<Double>> neighbors = findNeighbors(prediction);

		LogisticRegressionSmoother LR = new LogisticRegressionSmoother();
		LR.setPasses(passes);
		List<PrimitivePair> pairs = Lists.newArrayList();

		for (Entry<Double> entry : neighbors) {
			pairs.add(new PrimitivePair(entry.position.getValue(0), entry.value));
		}
		LR.train(pairs);
		return LR.smooth(prediction);
	}

	@Override
	public LocalLogisticRegressionSmoother getSmoother() {
		return this;
	}

	public LocalLogisticRegressionSmoother setPasses(int passes) {
		checkArgument(passes > 0, "passes must be positive, given: %s", passes);
		this.passes = passes;
		return this;
	}
}
