/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.evaluation;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import cern.colt.Arrays;

import com.parallax.ml.util.MLUtils;
import com.parallax.ml.util.pair.FirstDescendingComparator;
import com.parallax.ml.util.pair.PrimitivePair;

/**
 * stores a matrix encoding the costs of mistakes
 */

/**
 * The Class CostMatrix.
 */
public class CostMatrix {

	/** The misclassification costs associated with a predictive problem */
	private final double[][] costs;

	/** The number of label classes being considered. */
	private final int dims;

	/**
	 * Instantiates a new cost matrix.
	 * 
	 * @param costs
	 *            the costs
	 */
	public CostMatrix(double[][] costs) {
		this.costs = costs;
		this.dims = costs.length;
	}

	/**
	 * Compute total cost the total misclassification cost of a given a true and
	 * predicted assignment of probabilities to label classes
	 * 
	 * @param prediction
	 *            the predicted probability distribution on labels
	 * @param actual
	 *            the actual probability distribution on labels
	 * @return the cost of this predictive assignment
	 */
	public double computeTotalCost(double[] prediction, double[] actual) {
		// given a particular
		double cost = 0.0;
		for (int i = 0; i < this.dims; i++) {
			for (int j = 0; j < this.dims; j++) {
				cost += actual[i] * prediction[j] * costs[i][j];
			}
		}
		return cost;
	}

	/**
	 * Compute total cost the total misclassification cost of a given a true and
	 * predicted assignment of probabilities to label classes
	 * 
	 * @param prediction
	 *            a hard predicted assignment label to a single class
	 * @param actual
	 *            the actual probability distribution on labels
	 * @return the cost of this predictive assignment
	 */
	public double computeTotalCost(int prediction, double[] actual) {
		double[] predictions = new double[this.dims];
		predictions[prediction] = 1.0;
		return computeTotalCost(predictions, actual);
	}

	/**
	 * Compute total cost the total misclassification cost of a given a true and
	 * predicted assignment of probabilities to label classes
	 * 
	 * @param prediction
	 *            the predicted probability distribution on labels
	 * @param actual
	 *            the hard actual assignment of a label to a single class
	 * @return the cost of this predictive assignment
	 */
	public double computeTotalCost(double[] prediction, int actual) {
		double[] actuals = new double[this.dims];
		actuals[actual] = 1.0;
		return computeTotalCost(prediction, actuals);
	}

	/**
	 * Compute the costs made with respect to a particular input class
	 * 
	 * @param prediction
	 *            the predicted probability distribution on labels
	 * @param actual
	 *            the label class of interest
	 * @return the cost of this predictive assignment wrt the class of interest
	 */
	public double computeClassificationCost(double[] prediction, int dim) {
		checkArgument(dim < this.dims);
		double cost = 0.0;
		for (int i = 0; i < this.dims; i++) {
			if (i != dim) {
				cost += this.costs[i][dim] * prediction[i];
			}
		}
		return cost;

	}

	/**
	 * Computes the label assignment which minimizes misclassification costs
	 * 
	 * @param predictions
	 *            the raw class probabilities output by a predictive system
	 * @return the label class that minimizes prediction costs
	 */
	public int minCostClassification(double[] predictions) {
		int min = 0;
		double mincost = Double.MAX_VALUE;
		for (int i = 0; i < this.dims; i++) {
			double cost = this.computeClassificationCost(predictions, i);
			if (cost < mincost) {
				mincost = cost;
				min = i;
			}
		}
		return min;
	}

	/**
	 * Misclassification costs associated with the min cost class assignment
	 * 
	 * @param predictions
	 *            the predictions
	 * @return the double
	 */
	public double costOfMin(double[] predictions) {
		int choice = this.minCostClassification(predictions);
		return this.computeClassificationCost(predictions, choice);
	}

	/**
	 * Total test cost.
	 * 
	 * @param conf
	 *            the raw class probabilities output by a predictive system
	 * @return the costs of a min class assignment
	 */
	public double TotalTestCost(ConfusionMatrix conf) {
		double cost = 0;
		for (int i = 0; i < this.dims; i++) {
			for (int j = 0; j > this.dims; j++) {
				cost += costs[j][i] * conf.getMatrix()[i][j];
			}
		}
		return cost;
	}

	/**
	 * Actual loss; the loss incurred with the most likely ACTUAL label
	 * 
	 * @param prediction
	 *            the predicted probability distribution on labels
	 * @param actual
	 *            the actual probability distribution on labels
	 * @return the cost incurred by this prediction label assignment
	 */
	public double actualLoss(double[] predictions, double[] labels) {
		return costOfClassification(MLUtils.maxIndex(predictions),
				MLUtils.maxIndex(labels));
	}

	/**
	 * Cost of classification.
	 * 
	 * @param prediction
	 *            hard predicted class assignment
	 * @param label
	 *            hard actual class assignment
	 * @return the misclassification costs associated with this prediction /
	 *         label pair
	 */
	public double costOfClassification(int prediction, int label) {
		return this.costs[label][prediction];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuilder buff = new StringBuilder("[");
		for (int i = 0; i < dims; i++)
			buff.append(Arrays.toString(costs[i]) + "\n");
		buff.deleteCharAt(buff.length() - 1);
		buff.append("]");
		return buff.toString();
	}

	/**
	 * computes the cost curve associated with a particular set of labels and
	 * predictions
	 * 
	 * @param labelsAndPredictions
	 *            series of labels and predictions
	 * @return the representing the X,Y values of the cost curve
	 */
	public List<PrimitivePair> costCurve(
			Collection<PrimitivePair> labelsAndPredictions) {
		List<PrimitivePair> out = new ArrayList<PrimitivePair>(
				labelsAndPredictions.size());
		ConfusionMatrix conf = new ConfusionMatrix(2, labelsAndPredictions);

		double costFN = costOfClassification(0, 1);
		double costFP = costOfClassification(1, 0);
		double FN = 1 - conf.computePrecision(1);
		double FP = 1 - conf.computePrecision(0);

		out.add(new PrimitivePair(0, FP));
		out.add(new PrimitivePair(1, FN));

		for (PrimitivePair p : labelsAndPredictions) {
			double first = p.second * costFN
					/ (p.second * costFN + (1 - p.second) * costFP);
			double second = FN * first + FP * (1 - first);
			out.add(new PrimitivePair(first, second));
		}
		Collections.sort(out, new FirstDescendingComparator());
		return out;
	}
}
