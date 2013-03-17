/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.util.Collection;

import com.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
/**
 * The Class UpdateableLogisticRegressionSmoother.
 */
public class UpdateableLogisticRegressionSmoother extends
		AbstractUpdateableSmoother<UpdateableLogisticRegressionSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6149770736980313282L;

	/** The lr. */
	private LogisticRegression lr;
	
	/** The number of passes used when performing batch training. */
	protected int passes = 30;

	/** The initial weight used for sgd-like updates. */
	protected double weight = 0.1d;

	/**
	 * Instantiates a new updateable logistic regression smoother.
	 */
	public UpdateableLogisticRegressionSmoother() {
		lr = new LogisticRegression(1, true);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.UpdateableSmoother#update(com
	 * .parallax.ml.util.pair.PrimitivePair)
	 */
	@Override
	public void update(PrimitivePair p) {
		BinaryClassificationInstance x = new BinaryClassificationInstance(1);
		x.setLabel(new BinaryClassificationTarget(p.getSecond()));
		x.addFeature(0, p.getFirst());
		lr.update(x);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		BinaryClassificationInstance x = new BinaryClassificationInstance(1);
		x.addFeature(0, prediction);
		return lr.predict(x).getValue();
	}

	/**
	 * Gets the passes.
	 * 
	 * @return the passes
	 */
	public int getPasses() {
		return passes;
	}

	/**
	 * Gets the weight.
	 * 
	 * @return the weight
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * Sets the passes.
	 *
	 * @param passes the passes
	 * @return the updateable logistic regression smoother
	 */
	public UpdateableLogisticRegressionSmoother setPasses(int passes) {
		this.passes = passes;
		return this;
	}

	/**
	 * Sets the weight.
	 *
	 * @param newWeight the new weight
	 * @return the updateable logistic regression smoother
	 */
	public UpdateableLogisticRegressionSmoother setWeight(double newWeight) {
		this.weight = newWeight;
		return this;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.smoother.AbstractUpdateableSmoother#train(java.util.Collection)
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		for (int pass = 0; pass < passes; pass++) {
			for (PrimitivePair p : input)
				update(p);
			setWeight(getWeight() / 1.1);
		}
	}
}
