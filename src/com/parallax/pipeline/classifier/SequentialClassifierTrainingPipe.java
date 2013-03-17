/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.classifier;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.classifier.UpdateableClassifier;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * A pipe that trains an {@link UpdateableClassifier} sequentially as examples
 * are passed through the pipe. the output of the pipe is a performance
 * estmiate.
 * 
 * @param <C>
 *            the type of classifier used.
 */
public class SequentialClassifierTrainingPipe<C extends UpdateableClassifier<C>>
		extends AbstractPipe<BinaryClassificationInstance, OnlineEvaluation> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6934748792163455570L;

	/** The Constant CACHE_SIZE; what memory should the evaluation have? */
	private final static int CACHE_SIZE = 50000;

	/** The model to be trained */
	private UpdateableClassifier<C> model;

	/** The evaluator used for performance reporting */
	private transient OnlineEvaluation evaluator;

	/**
	 * Instantiates a new sequential classifier training pipe.
	 * 
	 * @param model
	 *            the model to be trained
	 */
	public SequentialClassifierTrainingPipe(UpdateableClassifier<C> model) {
		this(model, CACHE_SIZE);
	}

	/**
	 * Instantiates a new sequential classifier training pipe.
	 * 
	 * @param model
	 *            the model to be trained
	 * @param cacheWindow
	 *            how far back shoudl the evaluation look?
	 */
	public SequentialClassifierTrainingPipe(UpdateableClassifier<C> model,
			int cacheWindow) {
		super();
		this.model = model;
		this.evaluator = new OnlineEvaluation(cacheWindow);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<SequentialClassifierTrainingPipe<C>>() {
		}.getType();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<OnlineEvaluation> operate(
			Context<BinaryClassificationInstance> context) {
		BinaryClassificationInstance inst = context.getData();
		if (null != inst.getLabel()) {
			BinaryClassificationTarget target = model.predict(inst);
			evaluator.add(inst.getLabel().getValue(), target.getValue());
			model.update(inst);
		}
		return Context.createContext(context, evaluator);

	}
}
