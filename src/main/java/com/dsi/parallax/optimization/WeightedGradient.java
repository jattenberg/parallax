/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.util.ValueScaling;

public class WeightedGradient implements Iterable<Integer>, LinearVector {

	private final LinearVector gradientVector;
	private final double weight;
	private final double loss;

	/**
	 * @param weight
	 * @param gradientVector
	 * @param loss
	 */
	public WeightedGradient(double weight, LinearVector gradientVector,
			double loss) {
		checkArgument(weight >= 0,
				"gradient weight must be non-negative, given: %s", weight);
		this.weight = weight;
		this.gradientVector = gradientVector;

		this.loss = loss;
	}

	public WeightedGradient(double weight, LinearVector gradientVector) {
		this(weight, gradientVector, 0);
	}

	/**
	 * @param weight
	 * @param gradientVector
	 */
	public LinearVector getGradientVector() {
		return gradientVector;
	}

	public double getWeight() {
		return weight;
	}

	public double getLoss() {
		return loss;
	}

	@Override
	public Iterator<Integer> iterator() {
		return gradientVector.iterator();
	}

	public double getValue(int index) {
		return gradientVector.getValue(index);
	}

	public void updateValue(int index, double value) {
		gradientVector.updateValue(index, value);
	}

	public void resetValue(int index, double value) {
		gradientVector.resetValue(index, value);
	}

	public void delete(int index) {
		gradientVector.delete(index);
	}

	public Set<Integer> getFeatureIndicies() {
		return gradientVector.getFeatureIndicies();
	}

	public int size() {
		return gradientVector.size();
	}

	public void initW(double param) {
		gradientVector.initW(param);
	}

	public double[] getW() {
		return gradientVector.getW();
	}

	public double[] getWbias() {
		return gradientVector.getWbias();
	}

	public double[] getWbias(int biasTerms) {
		return gradientVector.getWbias(biasTerms);
	}

	public void setW(double[] W) {
		gradientVector.setW(W);
	}

	public void setW(List<Double> W) {
		gradientVector.setW(W);
	}

	public double LPNorm(double p) {
		return gradientVector.LPNorm(p);
	}

	public double L0Norm() {
		return gradientVector.L0Norm();
	}

	public double L1Norm() {
		return gradientVector.L1Norm();
	}

	public double L2Norm() {
		return gradientVector.L2Norm();
	}

	public double LInfinityNorm() {
		return gradientVector.LInfinityNorm();
	}

	public WeightedGradient timesEquals(double value) {
		gradientVector.timesEquals(value);
		return this;
	}

	public WeightedGradient plusEquals(double value) {
		gradientVector.plusEquals(value);
		return this;
	}

	public WeightedGradient minusEquals(double value) {
		gradientVector.minusEquals(value);
		return this;
	}

	public WeightedGradient plusEquals(LinearVector vect) {
		gradientVector.plusEquals(vect);
		return this;
	}

	public WeightedGradient minusEquals(LinearVector vect) {
		gradientVector.minusEquals(vect);
		return this;
	}

	public WeightedGradient plusEqualsVectorTimes(LinearVector vect,
			double factor) {
		gradientVector.plusEqualsVectorTimes(vect, factor);
		return this;
	}

	public WeightedGradient minusEqualsVectorTimes(LinearVector vect,
			double factor) {
		gradientVector.minusEqualsVectorTimes(vect, factor);
		return this;
	}

	public void absNormalize() {
		gradientVector.absNormalize();
	}

	@Override
	public LinearVector times(double value) {
		return gradientVector.times(value);
	}

	@Override
	public LinearVector plus(double value) {
		return gradientVector.plus(value);
	}

	@Override
	public LinearVector plus(LinearVector vect) {
		return gradientVector.plus(vect);
	}

	@Override
	public LinearVector minus(double value) {
		return gradientVector.minus(value);
	}

	@Override
	public LinearVector minus(LinearVector vect) {
		return gradientVector.minus(vect);
	}

	@Override
	public LinearVector plusVectorTimes(LinearVector vect, double factor) {
		return gradientVector.plusVectorTimes(vect, factor);
	}

	@Override
	public LinearVector minusVectorTimes(LinearVector vect, double factor) {
		return gradientVector.minusVectorTimes(vect, factor);
	}

	@Override
	public double dot(LinearVector vect) {
		return gradientVector.dot(vect);
	}

	@Override
	public double dot(LinearVector vect, ValueScaling scale) {
		return gradientVector.dot(vect, scale);
	}

}
