/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.supervisedfeture;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;
import com.parallax.ml.classifier.Classifier;
import com.parallax.ml.classifier.ClassifierBuilder;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.pipeline.AbstractAccumulatingPipe;
import com.parallax.pipeline.Context;

/**
 * a pipe component that adds an additional feature to examples based on the
 * prediction of a model, eg, one may wish to include the prediction of a
 * logistic regression model in the training of a higher variance decision tree
 * model.
 * 
 * @author jattenberg
 */
public class ClassifierPredictionFeaturePipe
		extends
		AbstractAccumulatingPipe<BinaryClassificationInstance, BinaryClassificationInstance> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -4839340589330084182L;

	/** The builder for the desired classifier type */
	private final ClassifierBuilder<?, ?> builder;

	/** The model used to make prediction features */
	private Classifier<?> model;

	/**
	 * Instantiates a new classifier prediction feature pipe.
	 * 
	 * @param builder
	 *            The builder for the desired classifier type
	 * @param toConsider
	 *            How many features to consider when training the initial
	 *            classifier
	 */
	public ClassifierPredictionFeaturePipe(ClassifierBuilder<?, ?> builder,
			int toConsider) {
		super(toConsider);
		this.builder = builder;
	}

	/**
	 * Instantiates a new classifier prediction feature pipe. Slurps all
	 * available examples in the pipeline when training the component classifier
	 * 
	 * @param builder
	 *            The builder for the desired classifier type
	 */
	public ClassifierPredictionFeaturePipe(ClassifierBuilder<?, ?> builder) {
		this(builder, -1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<ClassifierPredictionFeaturePipe>() {
		}.getType();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractAccumulatingPipe#operate(com.parallax.pipeline
	 * .Context)
	 */
	@Override
	protected Context<BinaryClassificationInstance> operate(
			Context<BinaryClassificationInstance> context) {
		BinaryClassificationInstance input = context.getData();
		BinaryClassificationTarget prediction = model.predict(input);
		LinearVector vect = LinearVectorFactory.copyToLength(input,
				input.getDimension() + 1);

		BinaryClassificationInstance out = new BinaryClassificationInstance(
				vect, input);
		out.resetValue(input.getDimension(), prediction.getValue());

		return Context.createContext(context, out);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractAccumulatingPipe#batchProcess(java.util
	 * .List)
	 */
	@Override
	protected void batchProcess(
			List<Context<BinaryClassificationInstance>> infoList) {
		Iterator<BinaryClassificationInstance> iter = Iterators.transform(
				Iterators.peekingIterator(infoList.iterator()), uncontextifier);
		int bins = Iterators.peekingIterator(infoList.iterator()).peek()
				.getData().getDimension();
		BinaryClassificationInstances insts = new BinaryClassificationInstances(
				bins);

		while (iter.hasNext())
			insts.addInstance(iter.next());

		model = builder.build();
		model.train(insts);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#isTrained()
	 */
	@Override
	public boolean isTrained() {
		return model != null;
	}
}
