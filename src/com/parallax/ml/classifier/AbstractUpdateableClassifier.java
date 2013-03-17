/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.classifier.smoother.UpdateableSmoother;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.pair.PrimitivePair;

// TODO: Auto-generated Javadoc
//TODO: apply more general annealing techniques used in the Optimization project
/**
 * base class for updateable machine learning models; models that can
 * incrementally incorporate additional information.
 * 
 * @param <C>
 *            the concrete type of updateable classifier. Used for method
 *            chaining.
 * @author jattenberg
 */
public abstract class AbstractUpdateableClassifier<C extends AbstractUpdateableClassifier<C>>
		extends AbstractClassifier<C> implements UpdateableClassifier<C> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6241246130691485103L;

	
	/** The number of passes used when performing batch training. */
	protected int passes = 30;

	/**
	 * Instantiates a new abstract updateable classifier.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected AbstractUpdateableClassifier(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Gets the passes; the number of passes used when performing batch
	 * training.
	 * 
	 * @return the passes
	 */
	public int getPasses() {
		return passes;
	}

	/**
	 * Sets the passes; number of passes used when performing batch training
	 * must be > 0.
	 * 
	 * @param passes
	 *            the passes
	 * @return the model itself, used for method chaining
	 */
	public C setPasses(int passes) {
		checkArgument(passes > 0, "passes must be greater than 0. input: %d",
				passes);
		this.passes = passes;
		return model;
	}

	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#modelTrain(com.parallax.ml.instance.Instances
	 * )
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances) {
		for (int i = 0; i < passes; i++) {
			for (Instance<BinaryClassificationTarget> inst : instances) {
				updateModel(inst);
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>> void update(I x) {
		trainSmoother(x);
		updateModel(x);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(com.parallax.ml
	 * .instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void update(
			I instst) {

		// TODO:
		for (Instance<BinaryClassificationTarget> inst : instst) {
			update(inst);
		}

	}

	/**
	 * updates the predictive component of the model.
	 * 
	 * @param <I>
	 *            the generic type
	 * @param instst
	 *            the instst
	 */
	public abstract <I extends Instance<BinaryClassificationTarget>> void updateModel(
			I instst);

	/**
	 * Update model.
	 * 
	 * @param <I>
	 *            the generic type
	 * @param instances
	 *            the instances
	 */
	public abstract <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void updateModel(
			I instances);

	/**
	 * Train smoother used for transforming raw classifier scores into
	 * probability estimates. updateable models use updateable smoother that
	 * can incrementally incorporate new information.
	 * 
	 * @param <I>
	 *            the type of instance used for training
	 * @param inst
	 *            the training dat used for improving the smoother
	 */
	protected <I extends Instance<BinaryClassificationTarget>> void trainSmoother(
			I inst) {
		if (null == smoother) {
			smoother = smootherType.getSmoother();
		} else {
			double prediction = regress(inst);
			double label = inst.getLabel().getValue();
			((UpdateableSmoother<?>) smoother).update(new PrimitivePair(
					prediction, label));
		}
	}

	/**
	 * adds a single instance into a instance collection
	 * 
	 * @param instance
	 *            to be added to a collection
	 * @return instances containing the supplied instance
	 */
	@SuppressWarnings("unchecked")
	protected static <I extends Instance<BinaryClassificationTarget>, J extends Instances<I>> J collect(
			I instance) {
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				instance.getDimension());
		insts.addInstance(new BinaryClassificationInstance(instance, instance
				.getLabel()));
		return (J) insts;
	}

	/**
	 * adds a iterator on instances into a instance collection
	 * 
	 * @param instances
	 *            to be added to a collection
	 * @return instances containing the supplied instance
	 */
	@SuppressWarnings("unchecked")
	protected static <I extends Instance<BinaryClassificationTarget>, J extends Instances<I>> J collect(
			Iterable<I> instances, int dimension) {
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				dimension);
		for (I instance : instances) {
			insts.addInstance(new BinaryClassificationInstance(instance,
					instance.getLabel()));
		}
		return (J) insts;
	}
}
