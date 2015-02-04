/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.util.ValueScaling;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;

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

	@Override
	public double getValue(int index) {
		return gradientVector.getValue(index);
	}

	@Override
	public void updateValue(int index, double value) {
		gradientVector.updateValue(index, value);
	}

	@Override
	public void resetValue(int index, double value) {
		gradientVector.resetValue(index, value);
	}

	@Override
	public void delete(int index) {
		gradientVector.delete(index);
	}

	@Override
	public Set<Integer> getFeatureIndicies() {
		return gradientVector.getFeatureIndicies();
	}

	@Override
	public int size() {
		return gradientVector.size();
	}

	@Override
	public void initW(double param) {
		gradientVector.initW(param);
	}

	@Override
	public double[] getW() {
		return gradientVector.getW();
	}

	@Override
	public double[] getWbias() {
		return gradientVector.getWbias();
	}

	@Override
	public double[] getWbias(int biasTerms) {
		return gradientVector.getWbias(biasTerms);
	}

	@Override
	public void setW(double[] W) {
		gradientVector.setW(W);
	}

	@Override
	public void setW(List<Double> W) {
		gradientVector.setW(W);
	}

	@Override
	public double LPNorm(double p) {
		return gradientVector.LPNorm(p);
	}

	@Override
	public double L0Norm() {
		return gradientVector.L0Norm();
	}

	@Override
	public double L1Norm() {
		return gradientVector.L1Norm();
	}

	@Override
	public double L2Norm() {
		return gradientVector.L2Norm();
	}

	@Override
	public double LInfinityNorm() {
		return gradientVector.LInfinityNorm();
	}

	@Override
	public WeightedGradient timesEquals(double value) {
		gradientVector.timesEquals(value);
		return this;
	}

	@Override
	public WeightedGradient plusEquals(double value) {
		gradientVector.plusEquals(value);
		return this;
	}

	@Override
	public WeightedGradient minusEquals(double value) {
		gradientVector.minusEquals(value);
		return this;
	}

	@Override
	public WeightedGradient plusEquals(LinearVector vect) {
		gradientVector.plusEquals(vect);
		return this;
	}

	@Override
	public WeightedGradient minusEquals(LinearVector vect) {
		gradientVector.minusEquals(vect);
		return this;
	}

	@Override
	public WeightedGradient plusEqualsVectorTimes(LinearVector vect,
			double factor) {
		gradientVector.plusEqualsVectorTimes(vect, factor);
		return this;
	}

	@Override
	public WeightedGradient minusEqualsVectorTimes(LinearVector vect,
			double factor) {
		gradientVector.minusEqualsVectorTimes(vect, factor);
		return this;
	}

	@Override
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

    @Override
    public LinearVector times(LinearVector vect) {
        this.gradientVector.times(vect);
        return this;
    }

    @Override
    public WeightedGradient copy() {
        return new WeightedGradient(weight, gradientVector.copy());
    }

}
