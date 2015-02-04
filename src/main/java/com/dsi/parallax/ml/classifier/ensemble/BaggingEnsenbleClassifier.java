package com.dsi.parallax.ml.classifier.ensemble;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.google.common.collect.Lists;
import org.apache.commons.cli.ParseException;

import java.lang.reflect.InvocationTargetException;
import java.util.List;

public class BaggingEnsenbleClassifier extends
		AbstractEnsembleClassifier<BaggingEnsenbleClassifier> {

	private static final long serialVersionUID = -225551522026894871L;

	protected BaggingEnsenbleClassifier(int dimension, boolean bias) {
		super(dimension, bias);
	}

	@Override
	public BaggingEnsenbleClassifier initialize() {
		return super.initialize();
	}

	@Override
	protected List<Double> subPredictions(Instance<?> inst) {
		List<Double> out = Lists.newArrayList();
		for (int fold = 0; fold < numModels; fold++) {
			out.add(models.get(fold).predict(inst).getValue());
		}
		return out;
	}

	@Override
	protected double combine(List<Double> predictions) {
		double sum = 0.;
		for (int i = 0; i < predictions.size(); i++) {
			sum += predictions.get(i);
		}
		return sum / predictions.size();
	}

	@Override
	protected <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances) {
		for (int fold = 0; fold < numModels; fold++) {
			models.get(fold).train(instances.getBag().getBagInstances());
		}
	}

	@Override
	protected BaggingEnsenbleClassifier getModel() {
		return this;
	}

	public static void main(String[] args) throws IllegalArgumentException,
			SecurityException, ParseException, IllegalAccessException,
			InvocationTargetException, NoSuchMethodException {
		ClassifierEvaluation.evaluate(args, BaggingEnsenbleClassifier.class);
	}
}
