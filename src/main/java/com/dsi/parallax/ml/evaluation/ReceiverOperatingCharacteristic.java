/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.util.pair.SecondDescendingComparator;

// TODO: Auto-generated Javadoc
/**
 * The Class ReceiverOperatingCharacteristic.
 */
public class ReceiverOperatingCharacteristic {
	
	/** The examples. */
	private List<PrimitivePair> examples;
	
	/** The pos. */
	private double pos = 0;
	
	/** The neg. */
	private double neg = 0;
	
	/** The is sorted. */
	private boolean isSorted = false;

	/**
	 * Instantiates a new receiver operating characteristic.
	 */
	public ReceiverOperatingCharacteristic() {
		examples = new ArrayList<PrimitivePair>();
	}

	/**
	 * Adds the.
	 *
	 * @param label the label
	 * @param prediction the prediction
	 */
	public void add(BinaryClassificationTarget label, BinaryClassificationTarget prediction) {
		add(label.getValue(), prediction.getValue());
	}
	
	/**
	 * Adds the.
	 *
	 * @param label the label
	 * @param prediction the prediction
	 */
	public void add(double label, double prediction) {
		examples.add(new PrimitivePair(label, prediction));
		if (label > 0.5)
			pos++;
		else
			neg++;
		isSorted = false;
	}

	/**
	 * Roc.
	 *
	 * @return the double[][]
	 */
	public double[][] ROC() {
		if (!isSorted) {
			Collections.sort(examples, new ExCompare());
			isSorted = true;
		}
		// checked: examples are in sorted order.
		// pos and neg are correct

		List<PrimitivePair> curve = new ArrayList<PrimitivePair>();
		double plast = Double.NEGATIVE_INFINITY;
		double tp = 0;
		double fp = 0;

		for (int i = 0; i < examples.size(); i++) {
			PrimitivePair current = examples.get(i);
			if (current.second != plast) {
				curve.add(new PrimitivePair(fp / neg, tp / pos));
				plast = current.second;
			}
			if (current.first > 0.5) {
				// positive example
				tp++;
			} else {
				// negative example
				fp++;
			}
		}
		curve.add(new PrimitivePair(fp / neg, tp / pos));

		double[][] out = new double[curve.size()][2];
		for (int i = 0; i < curve.size(); i++) {
			out[i][0] = curve.get(i).second; //tpr
			out[i][1] = curve.get(i).first; //fpr
		}
		return out;

	}

	/**
	 * Brier score.
	 *
	 * @return the double
	 */
	public double brierScore() {
		double score = 0;
		for (PrimitivePair ex : examples)
			score += (ex.second - ex.first) * (ex.second - ex.first);
		return score / examples.size();
	}

	/**
	 * bins the predictions. looks at the average label compared to the median
	 * prediction for each bin. computes the brier score based on this
	 *
	 * @param bins the bins
	 * @return the double
	 */
	public double averagedBrierScore(int bins) {
		double score = 0;
		double predBins = Math.min(bins, examples.size());
		if (!isSorted) {
			Collections.sort(examples, new SecondDescendingComparator());
			isSorted = true;
		}

		double binWidth = 1. / predBins;
		double bottom = 0.;
		double top = bottom + binWidth;
		int ct = 0;
		for (int i = 0; i < predBins; i++) {
			double num = 0;
			double avgLabel = 0;
			while (ct < examples.size() && examples.get(ct).second <= top
					&& examples.get(ct).second >= bottom) {
				avgLabel += examples.get(ct).first;
				ct++;
				num++;
			}
			double medianscore = (bottom + top) / 2.;

			if (num > 0) {
				avgLabel /= num;

				score += (medianscore - avgLabel) * (medianscore - avgLabel);
			}
			top += binWidth;
			bottom += binWidth;
		}
		return score / predBins;
	}

	/**
	 * Binary auc.
	 *
	 * @return the double
	 */
	public double binaryAUC() {
		double[][] ROC = ROC();
		double area = 0.0;
		for (int i = 1; i < ROC.length; i++) {
			area += trapezoidArea(ROC[i - 1][0], ROC[i][0], ROC[i - 1][1],
					ROC[i][1]);
		}
		area += trapezoidArea(1, ROC[ROC.length - 1][0], 1,
				ROC[ROC.length - 1][1]);
		return area;
	}

	/**
	 * Trapezoid area.
	 *
	 * @param x1 the x1
	 * @param x2 the x2
	 * @param y1 the y1
	 * @param y2 the y2
	 * @return the double
	 */
	private double trapezoidArea(double x1, double x2, double y1, double y2) {
		double base = Math.abs(x1 - x2);
		double avgHeight = (y1 + y2) / 2.0;
		return base * avgHeight;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();
		double[][] out = this.ROC();
		for (int i = 0; i < out.length; i++) {
			buff.append(out[i][0] + "\t" + out[i][1] + "\n");
		}
		return buff.toString();
	}

	/**
	 * RO cvalue.
	 *
	 * @param <C> the generic type
	 * @param model the model
	 * @param insts the insts
	 * @param roc the roc
	 */
	public static <C extends Classifier<C>> void ROCvalue(C model,
			BinaryClassificationInstances insts,
			ReceiverOperatingCharacteristic roc) {
		for (BinaryClassificationInstance x : insts) {
			double pred = model.predict(x).getValue();
			double label = x.getLabel().getValue();
			roc.add(label, pred);
		}
	}

	/**
	 * Compute auc.
	 *
	 * @param labelsAndPredictions the labels and predictions
	 * @return the double
	 */
	public static double computeAUC(
			Collection<PrimitivePair> labelsAndPredictions) {
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();
		for (PrimitivePair p : labelsAndPredictions)
			roc.add(p.first, p.second);
		return roc.binaryAUC();
	}

	/**
	 * Compute brier score.
	 *
	 * @param labelsAndPredictions the labels and predictions
	 * @return the double
	 */
	public static double computeBrierScore(
			Collection<PrimitivePair> labelsAndPredictions) {
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();
		for (PrimitivePair p : labelsAndPredictions)
			roc.add(p.first, p.second);
		return roc.averagedBrierScore(25);
	}

	/**
	 * The Class ExCompare.
	 */
	private class ExCompare implements Comparator<PrimitivePair> {
		
		/* (non-Javadoc)
		 * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
		 */
		@Override
		public int compare(PrimitivePair l1, PrimitivePair l2) {

			if (l1.second > l2.second)
				return 1;
			else if (l1.second < l2.second)
				return -1;
			else
				return 0;
		}
	}
}
