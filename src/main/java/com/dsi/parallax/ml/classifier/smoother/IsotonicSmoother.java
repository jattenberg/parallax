package com.dsi.parallax.ml.classifier.smoother;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.dsi.parallax.ml.util.pair.FirstDescendingComparator;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.google.common.collect.Lists;

// TODO: Auto-generated Javadoc
/**
 * an isotonic regression model used for smoothing classifier scores into smooth
 * probability estimates.
 * 
 * @author jattenberg
 * 
 */
public class IsotonicSmoother extends AbstractSmoother<IsotonicSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8416578875484203682L;

	/** The weights. */
	private double[] cuts, values, weights;

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		int index = Arrays.binarySearch(cuts, prediction);
		if (index < 0) {
			return values[-index - 1];
		} else {
			return values[index + 1];
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.AbstractSmoother#train(java
	 * .util.Collection)
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		List<PrimitivePair> inputList = Lists.newArrayList(input);

		// Sort values according to prediction
		Collections.sort(inputList, new FirstDescendingComparator());

		// Initialize arrays
		values = new double[input.size()];
		weights = new double[input.size()];
		cuts = new double[input.size() - 1];
		int size = 0;
		values[0] = inputList.get(0).second;
		weights[0] = 1;
		for (int i = 1; i < inputList.size(); i++) {
			if (inputList.get(i).first > inputList.get(i - 1).first) {
				cuts[size] = (inputList.get(i).first + inputList.get(i - 1).first) / 2;
				size++;
			}
			values[size] += inputList.get(i).second;
			weights[size]++;
		}
		size++;

		// While there is a pair of adjacent violators
		boolean violators;
		do {
			violators = false;

			// Initialize arrays
			double[] tempValues = new double[size];
			double[] tempWeights = new double[size];
			double[] tempCuts = new double[size - 1];

			// Merge adjacent violators
			int newSize = 0;
			tempValues[0] = values[0];
			tempWeights[0] = weights[0];
			for (int j = 1; j < size; j++) {
				if (values[j] / weights[j] > tempValues[newSize]
						/ tempWeights[newSize]) {
					tempCuts[newSize] = cuts[j - 1];
					newSize++;
					tempValues[newSize] = values[j];
					tempWeights[newSize] = weights[j];
				} else {
					tempWeights[newSize] += weights[j];
					tempValues[newSize] += values[j];
					violators = true;
				}
			}
			newSize++;

			// Copy references
			values = tempValues;
			weights = tempWeights;
			cuts = tempCuts;
			size = newSize;

		} while (violators);

		// Compute actual predictions
		for (int i = 0; i < size; i++) {
			values[i] /= weights[i];
		}

	}

}
