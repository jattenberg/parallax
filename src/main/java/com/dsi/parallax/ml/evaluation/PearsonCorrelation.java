/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import java.util.ArrayList;
import java.util.List;

import com.dsi.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * The Class PearsonCorrelation.
 */
public class PearsonCorrelation {
	
	/** The values. */
	List<PrimitivePair> values;

	/**
	 * Instantiates a new pearson correlation.
	 */
	public PearsonCorrelation() {
		values = new ArrayList<PrimitivePair>();
	}

	/**
	 * Adds the values.
	 *
	 * @param one the one
	 * @param two the two
	 */
	public void addValues(double one, double two) {
		values.add(new PrimitivePair(one, two));
	}

	/**
	 * Compute correlation.
	 *
	 * @return the double
	 */
	public double computeCorrelation() {
		PrimitivePair means = findMeans();
		PrimitivePair stddevs = findStdDevs(means);
		double cov = findCov(means);

		return cov / (stddevs.getFirst() * stddevs.getSecond());
	}

	/**
	 * Find means.
	 *
	 * @return the primitive pair
	 */
	private PrimitivePair findMeans() {
		double mu1 = 0, mu2 = 0;
		for (PrimitivePair p : values) {
			mu1 += p.getFirst();
			mu2 += p.getSecond();
		}
		mu1 /= values.size();
		mu2 /= values.size();

		return new PrimitivePair(mu1, mu2);
	}

	/**
	 * Find std devs.
	 *
	 * @param means the means
	 * @return the primitive pair
	 */
	private PrimitivePair findStdDevs(PrimitivePair means) {
		double sigma1 = 0, sigma2 = 0;
		double mu1 = means.getFirst();
		double mu2 = means.getSecond();
		for (PrimitivePair p : values) {
			sigma1 += Math.pow(p.getFirst() - mu1, 2.0);
			sigma2 += Math.pow(p.getSecond() - mu2, 2.0);
		}
		return new PrimitivePair(Math.sqrt(sigma1), Math.sqrt(sigma2));
	}

	/**
	 * Find cov.
	 *
	 * @param means the means
	 * @return the double
	 */
	private double findCov(PrimitivePair means) {
		double cov = 0.0;
		double mu1 = means.getFirst();
		double mu2 = means.getSecond();
		for (PrimitivePair p : values) {
			cov += (p.getFirst() - mu1) * (p.getSecond() - mu2);
		}
		return cov;
	}
}
