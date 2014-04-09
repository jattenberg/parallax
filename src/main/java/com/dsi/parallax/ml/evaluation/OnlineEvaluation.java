/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import java.io.Serializable;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Queue;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * class for performing an online evaluation of a predictive model. uses a
 * bounded queue to store recent model predictions typical use would be to
 * 
 * 1. get example x 2. make prediction f(x) 3. add example and label add(y,
 * f(x)) and 4. occassionally evaluate model's performance
 * 
 * implementation isn't optimized for re-evaluation of performance metrics at
 * each instance. some stratifying is done internally to avoid all-positive or
 * all-negative sets. two queues are simply used, one for positive and one for
 * negative.
 * 
 * stratification is avoided for accuracy and briar scoring by observing the
 * stream's class ratio and correcting for this ratio.
 * 
 * @author josh
 * 
 */
public class OnlineEvaluation implements Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8610292246440862829L;

	/** The negative cache. */
	private Queue<PrimitivePair> positiveCache, negativeCache;

	/** The maxsize. */
	private int maxsize = 50000;

	/** The numneg. */
	private double numpos, numneg;

	/**
	 * Instantiates a new online evaluation.
	 */
	public OnlineEvaluation() {
		this(Integer.MAX_VALUE);
	}

	/**
	 * Instantiates a new online evaluation.
	 * 
	 * @param cacheSize
	 *            the cache size
	 */
	public OnlineEvaluation(int cacheSize) {
		maxsize = cacheSize;
		numpos = 0;
		numneg = 0;
		positiveCache = new LinkedList<PrimitivePair>() {
			private static final long serialVersionUID = 8554777579995774523L;

			@Override
			public boolean add(PrimitivePair p) {
				if (size() >= maxsize)
					remove();
				return super.add(p);
			}
		};
		negativeCache = new LinkedList<PrimitivePair>() {
			private static final long serialVersionUID = 8554777579995774523L;

			@Override
			public boolean add(PrimitivePair p) {
				if (size() >= maxsize)
					remove();
				return super.add(p);
			}
		};
	}

	/**
	 * Adds the.
	 * 
	 * @param label
	 *            the label
	 * @param prediction
	 *            the prediction
	 * @return true, if successful
	 */
	public boolean add(BinaryClassificationTarget label,
			BinaryClassificationTarget prediction) {
		return add(label.getValue(), prediction.getValue());
	}

	/**
	 * Adds the.
	 * 
	 * @param label
	 *            the label
	 * @param prediction
	 *            the prediction
	 * @return true, if successful
	 */
	public boolean add(double label, double prediction) {
		if (label > 0.5) {
			numpos++;
			return positiveCache.add(new PrimitivePair(label, prediction));
		} else {
			numneg++;
			return negativeCache.add(new PrimitivePair(label, prediction));
		}
	}

	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> boolean add(
			I insts, Classifier<?> classifier) {
		boolean success = true;
		for (Instance<BinaryClassificationTarget> inst : insts) {
			BinaryClassificationTarget label = inst.getLabel();
			BinaryClassificationTarget prediction = classifier.predict(inst);
			success &= add(label, prediction);
		}
		return success;
	}

	/**
	 * Compute auc.
	 * 
	 * @return the double
	 */
	public double computeAUC() {
		return ReceiverOperatingCharacteristic.computeAUC(mergeSets());
	}

	/**
	 * Compute brier score.
	 * 
	 * @return the double
	 */
	public double computeBrierScore() {
		return ReceiverOperatingCharacteristic.computeBrierScore(mergeSets());
	}

	/**
	 * Compute accuracy.
	 * 
	 * @return the double
	 */
	public double computeAccuracy() {
		double tot = numpos + numneg;
		return (numpos / tot) * ConfusionMatrix.computeAccuracy(positiveCache)
				+ (numneg / tot)
				* ConfusionMatrix.computeAccuracy(negativeCache);
	}

	/**
	 * Merge sets.
	 * 
	 * @return the collection
	 */
	private Collection<PrimitivePair> mergeSets() {
		Collection<PrimitivePair> tmp = new LinkedList<PrimitivePair>();
		tmp.addAll(positiveCache);
		tmp.addAll(negativeCache);
		return tmp;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuilder buff = new StringBuilder();
		buff.append("AUC: " + computeAUC() + "\n")
				.append("ACCY: " + computeAccuracy() + "\n")
				.append("BRIER: " + computeBrierScore() + "\n");

		return buff.toString();
	}

	public static <I extends Instances<? extends Instance<BinaryClassificationTarget>>> OnlineEvaluation evaluate(
			I insts, Classifier<?> classifier) {
		OnlineEvaluation eval = new OnlineEvaluation(insts.size() * 2);
		eval.add(insts, classifier);
		return eval;
	}
}
