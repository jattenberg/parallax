package com.dsi.parallax.ml.classifier.ensemble;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.List;

import com.dsi.parallax.ml.classifier.AbstractClassifier;
import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.classifier.trees.ID3Builder;
import com.dsi.parallax.ml.instance.Instance;
import com.google.common.collect.Lists;

public abstract class AbstractEnsembleClassifier<C extends AbstractEnsembleClassifier<C>>
		extends AbstractClassifier<C> {

	private static final long serialVersionUID = -4228375624968026920L;

	protected ClassifierBuilder<?, ?> builder;

	protected List<Classifier<?>> models;

	protected int numModels = 10;

	protected AbstractEnsembleClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		setClassifierBuilder(new ID3Builder(dimension, bias)); // some default
																// builder;
	}

	@Override
	public C initialize() {
		models = Lists.newArrayList();
		for (int i = 0; i < numModels; i++) {
			models.add(builder.build());
		}
		return model;
	}

	protected abstract List<Double> subPredictions(Instance<?> inst);

	protected abstract double combine(List<Double> predictions);

	@Override
	protected double regress(Instance<?> inst) {
		return combine(subPredictions(inst));
	}

	public C setClassifierBuilder(ClassifierBuilder<?, ?> builder) {
		checkArgument(
				builder.getDimension() == dimension - (bias ? 1 : 0),
				"sub models must match dimension of the ensemble method. input: %s, expected: %s",
				builder.getDimension(), dimension - (bias ? 1 : 0));
		checkArgument(
				builder.getBias() == bias,
				"sub models must match the bias term of the ensemble, input: %s, expected %s",
				builder.getBias(), bias);
		this.builder = builder;
		return model;
	}

	public ClassifierBuilder<?, ?> getClassifierBuilder() {
		return this.builder;
	}

	public C setNumModels(int numModels) {
		checkArgument(numModels > 0, "numModels must be postitive, given: %s",
				numModels);
		this.numModels = numModels;
		return model;
	}

	public int getNumModels() {
		return numModels;
	}
}
