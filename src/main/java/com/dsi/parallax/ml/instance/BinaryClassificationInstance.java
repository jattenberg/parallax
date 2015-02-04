/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.instance;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.util.ValueScaling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Class representing Instanzes with binary classification labels.
 */
public class BinaryClassificationInstance extends
		Instance<BinaryClassificationTarget> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6040292370882152602L;

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param label
	 *            applied to this instance.
	 * @param values
	 *            feature values describing the instance.
	 */
	public BinaryClassificationInstance(double label, double[] values) {
		this(new BinaryClassificationTarget(label), values);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param label
	 *            applied to this instance.
	 * @param values
	 *            feature values describing the instance.
	 */
	public BinaryClassificationInstance(BinaryClassificationTarget label,
			double[] values) {
		super(label, values);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param label
	 *            applied to this instance.
	 * @param values
	 *            feature values describing the instance.
	 */
	public BinaryClassificationInstance(BinaryClassificationTarget label,
			LinearVector values) {
		super(label, values);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param dimensions
	 *            number of dimensions in the instance.
	 * @param label
	 *            the label applied to this instance.
	 */
	public BinaryClassificationInstance(int dimensions,
			BinaryClassificationTarget label) {
		super(dimensions, label);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param dimensions
	 *            number of dimensions in the instance.
	 * @param weight
	 *            importance weight of the instance.
	 */
	public BinaryClassificationInstance(int dimensions, double weight) {
		super(dimensions, weight);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param dimensions
	 *            the number of dimensions in the instance.
	 */
	public BinaryClassificationInstance(int dimensions) {
		super(dimensions);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param vector
	 *            feature values describing the instance.
	 */
	public BinaryClassificationInstance(LinearVector vector) {
		super(vector);
	}

	/**
	 * Instantiates a new binary classification instance.
	 * 
	 * @param vector
	 *            feature values describing the instance.
	 */
	public BinaryClassificationInstance(LinearVector vector,
			BinaryClassificationTarget label) {
		super(vector);
		this.label = label;
	}

	/**
	 * copies the label, id, and frequency info from the instance but uses the
	 * input vector.
	 * 
	 * @param vector
	 *            feature values describing the instance.
	 * @param instance
	 *            the instance from where instance is to be copied.
	 */
	public BinaryClassificationInstance(LinearVector vector,
			BinaryClassificationInstance instance) {
		super(vector, instance);
	}

	/**
	 * To svn light format.
	 * 
	 * @see {@link <a href="http://svmlight.joachims.org/">Svmlight</a>}
	 * @return the string representation of the current instance in svmlight
	 *         format.
	 */
	public String toSVNLightFormat() {
		StringBuilder buff = new StringBuilder(
				""
						+ (MLUtils.floatingPointEquals(getLabel()
								.getValue(), 0) ? -1 : 1));
		List<PrimitivePair> featureValues = new ArrayList<PrimitivePair>();
		for (int x_i : this) {
			featureValues.add(new PrimitivePair(x_i + 1, getFeatureValue(
					ValueScaling.UNSCALED, x_i)));
		}
		Collections.sort(featureValues, new Comparator<PrimitivePair>() {
			@Override
			public int compare(PrimitivePair a, PrimitivePair b) {
				return a.getFirst() > b.getFirst() ? 1 : b.getFirst() > a
						.getFirst() ? -1 : 0;
			}
		});
		for (PrimitivePair p : featureValues) {
			int x_i = (int) p.getFirst();
			double y_i = p.getSecond();
			buff.append(" " + x_i + ":" + y_i);
		}
		return buff.toString();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#clone()
	 */
	@Override
	public BinaryClassificationInstance clone() {
		BinaryClassificationInstance out = new BinaryClassificationInstance(
				getDimension());
		out.label = this.label;
		out.setLabel(label);
		out.freq = freq;
		out.ID = ID;
		out.weight = weight;
		for (int x : this)
			out.addFeature(x, getFeatureValue(ValueScaling.UNSCALED, x));
		return out;
	}

	/**
	 * Instance representing the difference of the second instance from the
	 * first instance.
	 * 
	 * @param a
	 *            The first instance from which the difference is being
	 *            computed.
	 * @param b
	 *            The second instance being subtracted
	 * @return the instance representing the difference.
	 */
	public static BinaryClassificationInstance instanceFromDifference(
			BinaryClassificationInstance a, BinaryClassificationInstance b) {
		BinaryClassificationInstance out = new BinaryClassificationInstance(
				a.getDimension());
		for (int x_i : a)
			out.addFeature(x_i, a.getFeatureValue(ValueScaling.UNSCALED, x_i));
		for (int x_i : b)
			out.addFeature(x_i, -b.getFeatureValue(ValueScaling.UNSCALED, x_i));
		out.setID(a.ID + "-" + b.ID);
		return out;
	}

	/**
	 * Copy an instance.
	 * 
	 * @param in
	 *            instance to be copied
	 * @return a copy of the input.
	 */
	public static BinaryClassificationInstance copyInstance(
			BinaryClassificationInstance in) {
		return in.clone();
	}

	/**
	 * Copy all of the information of an instance, but replace a vector with the
	 * input vector.
	 * 
	 * @param in
	 *            instance who's information is to be copied.
	 * @param vector
	 *            new feature values describing the instance.
	 * @return new instance combining the input information.
	 */
	public static BinaryClassificationInstance copyInstanceChangeVector(
			BinaryClassificationInstance in, LinearVector vector) {
		BinaryClassificationInstance inst = in.clone();
		inst.vector = vector;
		return inst;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.instance.Instanze#cloneNewVector(com.parallax.ml.util
	 * .vector.LinearVector)
	 */
	@Override
	public Instance<BinaryClassificationTarget> cloneNewVector(LinearVector vec) {
		BinaryClassificationInstance inst = new BinaryClassificationInstance(
				vec);
		inst.label = label;
		inst.weight = weight;
		inst.freq = freq;
		inst.ID = ID;
		return inst;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#getWbias()
	 */
	@Override
	public double[] getWbias() {
		return vector.getWbias();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.vector.LinearVector#getWbias(int)
	 */
	@Override
	public double[] getWbias(int biasTerms) {
		return vector.getWbias(biasTerms);
	}

    @Override
    public BinaryClassificationInstance copy() {
        return new BinaryClassificationInstance(this.label, vector.copy());
    }

    @Override
    public BinaryClassificationInstance times(LinearVector vect) {
        this.vector.times(vect);
        return this;
    }
}
