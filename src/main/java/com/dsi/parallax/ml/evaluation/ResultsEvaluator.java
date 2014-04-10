/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import java.io.BufferedWriter;
import java.io.FileWriter;

// TODO: Auto-generated Javadoc
/*
 * TODO: binary precision @ recall curves
 * TODO: multiclass precision @ recall curves
 * TODO: binary AUC 
 * TODO: multiclass AUC
 * 
 * (multiclass 1 vs all of rest )
 * (multiclass 1 vs each of rest )
 * 
 * TODO: learning curves for each analysis?
 * TODO: learning curves can be a different class that has an instance of this. 
 */
class ResultPair {
	double[] prediction;
	double[] actual;

	ResultPair(double[] prediction, double[] actual) {
		this.prediction = prediction;
		this.actual = actual;
	}
}

class PrecisionRecall {
	double precision;
	double recall;

	PrecisionRecall(double precision, double recall) {
		this.precision = precision;
		this.recall = recall;
	}

	public double getP() {
		return this.precision;
	}

	public double getR() {
		return this.recall;
	}
}

class PRCurve {
	java.util.Vector<PrecisionRecall> PatR;

	PRCurve() {
		PatR = new java.util.Vector<PrecisionRecall>();
	}

	public void addPoint(PrecisionRecall point) {
		this.PatR.add(point);
	}

	public java.util.Vector<PrecisionRecall> getCurve() {
		return this.PatR;
	}

	public int getSize() {
		return this.PatR.size();
	}

	public PrecisionRecall[] getArray() {
		return (PrecisionRecall[]) PatR.toArray();
	}
}

class multiCurve {
	double[][] Xs;
	double[][] Ys;
	int num_bins;

	multiCurve(int num_bins) {
		this.num_bins = num_bins;
		Xs = new double[num_bins][];
		Ys = new double[num_bins][];
	}

	public void addPR(PrecisionRecall[] points, int dimension) {
		Xs[dimension] = new double[points.length];
		Ys[dimension] = new double[points.length];
		for (int i = 0; i < points.length; i++) {
			Xs[dimension][i] = points[i].getR();
			Ys[dimension][i] = points[i].getP();
		}
	}

	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();
		for (int i = 0; i < this.num_bins; i++) {
			for (int j = 0; j < this.Xs[i].length; j++) {
				buff.append(Xs[i][j] + "\t");
			}
			buff.append("\n");
			for (int j = 0; j < this.Xs[i].length; j++) {
				buff.append(Ys[i][j] + "\t");
			}
			buff.append("\n");
		}
		return buff.toString();
	}

	public void print() {
		System.out.print(this.toString());
	}

	public void print(String filename) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			writer.write(this.toString());
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}

/**
 * The Class ResultsEvaluator.
 */
public class ResultsEvaluator {

	/** The results. */
	java.util.Vector<ResultPair> results;

	/** The costmatrix. */
	CostMatrix costmatrix;

	/** The num classes. */
	int numClasses = 2;

	/** The roc. */
	ReceiverOperatingCharacteristic ROC;

	/**
	 * Instantiates a new results evaluator.
	 */
	public ResultsEvaluator() {
		this.results = new java.util.Vector<ResultPair>();
		this.costmatrix = new CostMatrix(new double[numClasses][numClasses]);
	}

	/**
	 * Instantiates a new results evaluator.
	 * 
	 * @param numClasses
	 *            the num classes
	 */
	public ResultsEvaluator(int numClasses) {
		this.numClasses = numClasses;
		this.results = new java.util.Vector<ResultPair>();
		this.costmatrix = new CostMatrix(new double[numClasses][numClasses]);
	}

	/**
	 * Adds the result.
	 * 
	 * @param prediction
	 *            the prediction
	 * @param actual
	 *            the actual
	 */
	public void addResult(double[] prediction, double[] actual) {
		// add two multinomails, a prediction and an actual distirbution
		this.results.add(new ResultPair(prediction, actual));
	}

	/**
	 * Adds the result hard.
	 * 
	 * @param prediction
	 *            the prediction
	 * @param actual
	 *            the actual
	 */
	public void addResultHard(int prediction, int actual) {
		// add a hard class assignment and a hard actual assignment
		double[] predictions = new double[numClasses];
		double[] actuals = new double[numClasses];

		predictions[prediction] = 1.0;
		actuals[actual] = 1.0;

		this.addResult(predictions, actuals);
	}

	/**
	 * Adds the result.
	 * 
	 * @param prediction
	 *            the prediction
	 * @param actual
	 *            the actual
	 */
	public void addResult(double[] prediction, int actual) {
		// add a multinomial prediction, and a hard class assignment
		double[] actuals = new double[numClasses];
		actuals[actual] = 1.0;
		this.addResult(prediction, actuals);
	}

	/**
	 * Precision at recall one vs one.
	 * 
	 * @param thresh
	 *            the thresh
	 * @param dim
	 *            the dim
	 * @return the precision recall
	 */
	public PrecisionRecall precisionAtRecallOneVsOne(double thresh, int dim) {
		// for a given threshhold, compute precision at recall for a given
		// confusion matrix
		ConfusionMatrix conf = new ConfusionMatrix(2);
		for (int i = 0; i < results.size(); i++) {
			int pred = 0;
			int label = 0;
			if (results.get(i).prediction[dim] > thresh) {
				pred = 1;
			}
			if (results.get(i).actual[dim] > thresh) {
				label = 1;
			}
			conf.addInfo(label, pred);
		}

		return new PrecisionRecall(conf.computePrecision(dim),
				conf.computeRecall(dim));
	}

	/**
	 * Precision at recall at dim.
	 * 
	 * @param dim
	 *            the dim
	 * @return the pR curve
	 */
	public PRCurve precisionAtRecallAtDim(int dim) {
		// chose 1 v 1 or 1 v all,
		// classify according to a particular threshold
		// use resulting confusion matrix to compute precision and recall

		PRCurve PatR = new PRCurve();
		for (double thresh = 0.0; thresh <= 1; thresh += 0.01) {

			PatR.addPoint(precisionAtRecallOneVsOne(thresh, dim));
		}
		return PatR;
	}

	/**
	 * Precision at recall.
	 * 
	 * @return the multi curve
	 */
	public multiCurve precisionAtRecall() {
		multiCurve mc = new multiCurve(this.numClasses);
		for (int i = 0; i < this.numClasses; i++) {
			PRCurve prc = this.precisionAtRecallAtDim(i);
			mc.addPR(prc.getArray(), i);
		}
		return mc;
	}

	/*
	 * private int maxIndex(double[] array) { int max = 0; double val =
	 * Double.MIN_VALUE; for(int i = 0; i < array.length; i++) { if(array[i] >
	 * val) { val = array[i]; max = i; } } return max; }
	 */
}
