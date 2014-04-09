/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.instance;

import static com.dsi.parallax.ml.util.MLUtils.hashDouble;
import static com.google.common.base.Preconditions.checkArgument;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import com.dsi.parallax.ml.target.Target;
import com.dsi.parallax.ml.util.pair.FirstDescendingComparator;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.ml.vector.util.ValueScaling;
import com.google.common.collect.Maps;

// TODO: Auto-generated Javadoc
/**
 * container class for machine learning examples.
 * 
 * @param <T>
 *            the generic type
 * @author josh
 */
public abstract class Instance<T extends Target> implements Serializable,
		LinearVector {

	/** The Constant serialVersionUID. */
	static final long serialVersionUID = 1L;

	/** The vec. */
	protected LinearVector vector;

	/** The label. */
	protected T label;

	/** The weight. */
	protected double weight = 1.0;

	/** The freq. */
	protected double freq = 1;

	/** The id. */
	protected String ID = "none";

	/** The max pos index. */
	protected int maxPosIndex = -1;

	/** The hashcode. */
	protected int hashcode = 0;

	/** The LIN fnorm. */
	protected double L0norm = -1, L1norm = -1, L2norm = -1, LINFnorm = -1; // lazily
																			// instantiated

	/**
	 * Instantiates a new instance.
	 * 
	 * @param dimensions
	 *            the dimensions
	 */
	public Instance(int dimensions) {
		this.vector = LinearVectorFactory.getVector(dimensions);
	}

	/**
	 * Instantiates a new instance.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param label
	 *            the label
	 */
	public Instance(int dimensions, T label) {
		this.label = label;
	}

	/**
	 * Instantiates a new instance.
	 * 
	 * @param label
	 *            the label
	 * @param values
	 *            the values
	 */
	public Instance(T label, double[] values) {
		this.vector = LinearVectorFactory.getVector(values);
		this.label = label;
	}

	/**
	 * Instantiates a new instance.
	 * 
	 * @param label
	 *            the label
	 * @param values
	 *            the values
	 */
	public Instance(T label, LinearVector values) {
		this.vector = values;
		this.label = label;
	}

	/**
	 * Instantiates a new instance.
	 * 
	 * @param dimensions
	 *            the dimensions
	 * @param weight
	 *            the weight
	 */
	public Instance(int dimensions, double weight) {
		this(dimensions);
		this.weight = weight;
	}

	/**
	 * Instantiates a new instance.
	 * 
	 * @param vector
	 *            the vector
	 */
	public Instance(LinearVector vector) {
		this.vector = vector;
	}

	/**
	 * copies the label, id, and frequency info from the instance but uses the
	 * input vector.
	 * 
	 * @param vector
	 *            the vector
	 * @param instance
	 *            the instance
	 */
	public Instance(LinearVector vector, Instance<T> instance) {
		this.vector = vector;
		this.ID = instance.ID;
		this.label = instance.label;
		this.freq = instance.freq;
	}

	/**
	 * copies the label, id, frequency and input vector info from the instance
	 * 
	 * 	@param instance
	 *  	         the instance
	 */
	public Instance(Instance<T> instance) {
		this.vector = instance.vector;
		this.ID = instance.ID;
		this.label = instance.label;
		this.freq = instance.freq;
	}

	/**
	 * Gets the dimension.
	 * 
	 * @return the dimension
	 */
	public int getDimension() {
		return this.vector.size();
	}

	/**
	 * Gets the id.
	 * 
	 * @return the id
	 */
	public String getID() {
		return this.ID;
	}

	/**
	 * Sets the id.
	 * 
	 * @param ID
	 *            the new id
	 */
	public void setID(String ID) {
		this.ID = ID;
	}

	/**
	 * Gets the feature values.
	 * 
	 * @return the feature values
	 */
	public LinearVector getFeatureValues() {
		return vector;
	}

	/**
	 * Sets the label.
	 * 
	 * @param label
	 *            the new label
	 */
	public void setLabel(T label) {
		checkArgument(label != null, "input label must not be null");
		this.label = label;
	}

	/**
	 * Gets the label.
	 * 
	 * @return the label
	 */
	public T getLabel() {
		return this.label;
	}

	/**
	 * Gets the freq.
	 * 
	 * @return the freq
	 */
	public double getFreq() {
		return freq;
	}

	/**
	 * Sets the freq.
	 * 
	 * @param freq
	 *            the new freq
	 */
	public void setFreq(double freq) {
		this.freq = freq;
	}

	/**
	 * Gets the max pos index.
	 * 
	 * @return the max pos index
	 */
	public int getMaxPosIndex() {
		return maxPosIndex;
	}

	/**
	 * Sets the max pos index.
	 * 
	 * @param maxPosIndex
	 *            the new max pos index
	 */
	public void setMaxPosIndex(int maxPosIndex) {
		this.maxPosIndex = maxPosIndex;
	}

	/**
	 * Sets the weight.
	 * 
	 * @param weight
	 *            the new weight
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}

	/**
	 * Gets the weight.
	 * 
	 * @return the weight
	 */
	public double getWeight() {
		return this.weight;
	}

	/**
	 * Adds the feature.
	 * 
	 * @param dim
	 *            the dim
	 * @param value
	 *            the value
	 */
	public void addFeature(int dim, double value) {
		vector.updateValue(dim, value);
		L0norm = -1;
		L2norm = -1;
		L1norm = -1;
		LINFnorm = -1;
	}

	/**
	 * Gets the sorted indices.
	 * 
	 * @return the sorted indices
	 */
	public Set<Integer> getSortedIndicies() {
		Set<Integer> out = new TreeSet<Integer>(new Comparator<Integer>() {

			@Override
			public int compare(Integer v1, Integer v2) {
				return v1 > v2 ? 1 : v2 > v1 ? -1 : 0;
			}

		});
		for (int index : this)
			out.add(index);
		return out;
	}

	/**
	 * Gets the feature value.
	 * 
	 * @param dimension
	 *            the dimension
	 * @return the feature value
	 */
	public double getFeatureValue(int dimension) {
		return getFeatureValue(ValueScaling.UNSCALED, dimension);
	}

	/**
	 * Gets the feature value.
	 * 
	 * @param scaling
	 *            the scaling
	 * @param dimension
	 *            the dimension
	 * @return the feature value
	 */
	public double getFeatureValue(ValueScaling scaling, int dimension) {
		return scaling.scaleValue(vector.getValue(dimension));
	}

	/**
	 * Prints the instance.
	 */
	public void printInstance() {
		System.out.println(this.toString());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		StringBuffer buff = new StringBuffer(label + "- ");

		for (Integer x : this) {
			double y = vector.getValue(x);
			buff.append(x + ":" + y + " ");
		}
		return buff.toString();
	}

	/**
	 * Sparse to dense.
	 * 
	 * @return the double[]
	 */
	public double[] sparseToDense() {
		return vector.getW();
	}

	/**
	 * Str inst.
	 * 
	 * @return the string
	 */
	public String strInst() {
		StringBuffer buff = new StringBuffer();
		List<PrimitivePair> pairs = new ArrayList<PrimitivePair>();
		for (int x : this) {
			double y = getFeatureValue(ValueScaling.UNSCALED, x);
			pairs.add(new PrimitivePair(x, y));
		}

		Collections.sort(pairs, new FirstDescendingComparator());
		for (PrimitivePair pair : pairs) {
			buff.append((int) pair.first + ":" + (int) pair.second + "\t");
		}

		buff.append(this.vector.size() + ":" + this.label);
		return buff.toString();

	}

	/**
	 * To vw format.
	 * 
	 * @return the string
	 */
	public String toVWFormat() {
		StringBuilder buff = new StringBuilder(label + " |features ");
		for (int x : this)
			buff.append(x + ":" + getFeatureValue(x) + " ");
		buff.append("const:.01");

		return buff.toString();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Iterable#iterator()
	 */
	@Override
	public Iterator<Integer> iterator() {
		return vector.iterator();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		int result = hashcode;
		if (result == 0) {
			result = 17;
			result = 31 * result + ID.hashCode();
			result = 31 * result + label.hashCode();
			result = 31 * result + vector.size();
			for (int x : this) {
				result = 31 * result + x;
				result = 31 * result
						+ hashDouble(getFeatureValue(ValueScaling.UNSCALED, x));
			}
			hashcode = result;
		}
		return result;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		else if (!(o instanceof Instance))
			return false;
		else {
			Instance<?> x = (Instance<?>) o;
			boolean check = (ID == x.ID && getDimension() == x.getDimension() && label == x.label);
			if (!check)
				return false;
			else {
				if ((label != null && x.label == null)
						|| (label == null && x.label != null)
						|| !label.equals(x.label))
					return false;

				if (vector.size() != x.vector.size())
					return false;
				for (int y : this)
					if (getFeatureValue(y) != x.getFeatureValue(y))
						return false;
				for (int y : x)
					if (getFeatureValue(y) != x.getFeatureValue(y))
						return false;

				return true;
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#size()
	 */
	public int size() {
		return vector.size();
	}

	/**
	 * To map.
	 * 
	 * @return the map< integer,? extends number>
	 */
	public Map<Integer, ? extends Number> toMap() {
		Map<Integer, Double> out = Maps.newHashMap();
		for (int x_i : this)
			out.put(x_i, getFeatureValue(x_i));
		return out;
	}

	/**
	 * Inner product.
	 * 
	 * @param x
	 *            the x
	 * @param y
	 *            the y
	 * @return the double
	 */
	public static final double innerProduct(Instance<?> x, Instance<?> y) {
		double tot = 0;
		Set<Integer> seen = new HashSet<Integer>();
		for (int i : x) {
			tot += x.getFeatureValue(i) * y.getFeatureValue(i);
			seen.add(i);
		}
		for (int i : y) {
			if (!seen.contains(i))
				tot += y.getFeatureValue(i) * x.getFeatureValue(i);
		}
		return tot;
	}

	/**
	 * Compute l ndist.
	 * 
	 * @param x
	 *            the x
	 * @param y
	 *            the y
	 * @param n2
	 *            the n2
	 * @return the double
	 */
	public static double computeLNdist(Instance<?> x, Instance<?> y, double n2) {
		double tot = 0;
		Set<Integer> indexSet = new HashSet<Integer>();
		for (int i : x) {
			indexSet.add(i);
			tot += Math.pow(
					Math.abs(x.getFeatureValue(i) - y.getFeatureValue(i)), n2);
		}
		for (int i : y) {
			if (!indexSet.contains(i))
				tot += Math.pow(
						Math.abs(x.getFeatureValue(i) - y.getFeatureValue(i)),
						n2);
		}
		return Math.pow(tot, 1. / n2);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#getValue(int)
	 */
	public double getValue(int index) {
		return vector.getValue(index);
	}

	/**
	 * takes whatever value is at index and increments by value.
	 * 
	 * @param index
	 *            the index
	 * @param value
	 *            the value
	 */
	public void updateValue(int index, double value) {
		vector.updateValue(index, value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#resetValue(int, double)
	 */
	public void resetValue(int index, double value) {
		vector.resetValue(index, value);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#delete(int)
	 */
	public void delete(int index) {
		vector.delete(index);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#getFeatureIndicies()
	 */
	public Set<Integer> getFeatureIndicies() {
		return vector.getFeatureIndicies();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#initW(double)
	 */
	public void initW(double param) {
		vector.initW(param);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#getW()
	 */
	public double[] getW() {
		return vector.getW();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#setW(double[])
	 */
	public void setW(double[] W) {
		vector.setW(W);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#setW(java.util.List)
	 */
	public void setW(List<Double> W) {
		vector.setW(W);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#LPNorm(double)
	 */
	public double LPNorm(double p) {
		return vector.LPNorm(p);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#L0Norm()
	 */
	public double L0Norm() {
		return vector.L0Norm();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#L1Norm()
	 */
	public double L1Norm() {
		return vector.L1Norm();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#L2Norm()
	 */
	public double L2Norm() {
		return vector.L2Norm();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#LInfinityNorm()
	 */
	public double LInfinityNorm() {
		return vector.LInfinityNorm();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#timesEquals(double)
	 */
	public LinearVector timesEquals(double value) {
		vector.timesEquals(value);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#plusEquals(double)
	 */
	public Instance<T> plusEquals(double value) {
		vector.plusEquals(value);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#minusEquals(double)
	 */
	public Instance<T> minusEquals(double value) {
		vector.minusEquals(value);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.vector.LinearVector#plusEquals(com.parallax.ml.util
	 * .vector.LinearVector)
	 */
	public Instance<T> plusEquals(LinearVector vect) {
		vector.plusEquals(vect);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.vector.LinearVector#minusEquals(com.parallax.ml.
	 * util.vector.LinearVector)
	 */
	public Instance<T> minusEquals(LinearVector vect) {
		vector.minusEquals(vect);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.vector.LinearVector#plusEqualsVectorTimes(com.parallax
	 * .ml.util.vector.LinearVector, double)
	 */
	public Instance<T> plusEqualsVectorTimes(LinearVector vect, double factor) {
		vector.plusEqualsVectorTimes(vect, factor);
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.vector.LinearVector#minusEqualsVectorTimes(com.parallax
	 * .ml.util.vector.LinearVector, double)
	 */
	public Instance<T> minusEqualsVectorTimes(LinearVector vect, double factor) {
		vector.minusEqualsVectorTimes(vect, factor);
		return this;
	}

	/**
	 * normalize by the L1 norm.
	 */
	public void absNormalize() {
		vector.absNormalize();
	}

	/**
	 * Clone new vector.
	 * 
	 * @param vec
	 *            the vec
	 * @return the instanze
	 */
	public abstract Instance<T> cloneNewVector(LinearVector vec);

	@Override
	public LinearVector times(double value) {
		return vector.times(value);
	}

	@Override
	public LinearVector plus(double value) {
		return vector.plus(value);
	}

	@Override
	public LinearVector plus(LinearVector vect) {
		return vector.plus(vect);
	}

	@Override
	public LinearVector minus(double value) {
		return vector.minus(value);
	}

	@Override
	public LinearVector minus(LinearVector vect) {
		return vector.minus(vect);
	}

	@Override
	public LinearVector plusVectorTimes(LinearVector vect, double factor) {
		return vector.plusVectorTimes(vect, factor);
	}

	@Override
	public LinearVector minusVectorTimes(LinearVector vect, double factor) {
		return vector.minusVectorTimes(vect, factor);
	}

	@Override
	public double dot(LinearVector vect) {
		return vector.dot(vect);
	}

	@Override
	public double dot(LinearVector vect, ValueScaling scale) {
		return vector.dot(vect, scale);
	}
}
