/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import java.util.Set;

import org.apache.log4j.Logger;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.ObjectiveScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.google.common.collect.Sets;

/**
 * selects the best Classifier (using default configurations) for input data.
 * 
 * @author jattenberg
 */
public class ClassifierSelection {

	/** logger to record the decision process */
	private static final Logger LOGGER = Logger
			.getLogger(ClassifierSelection.class);

	/** The set candidate classifiers to be considered */
	private final Set<Classifiers> candidateClassifiers;

	/**
	 * The Constant DEFAULT_FOLDS; how many folds of cross validation should be
	 * performed unless told otherwise?
	 */
	private static final int DEFAULT_FOLDS = 5;

	/** how many folds of cross validation should be performed? */
	private final int folds;

	/** The evaluator used for performing multi-threaded cross validation */
	private final FoldEvaluator evaluator;

	/**
	 * Instantiates a new classifier selection.
	 * 
	 * @param candidateClassifiers
	 *            the set of {@link Classifiers} to consider
	 */
	public ClassifierSelection(Set<Classifiers> candidateClassifiers) {
		this.candidateClassifiers = candidateClassifiers;
		folds = DEFAULT_FOLDS;
		evaluator = new FoldEvaluator(folds);
	}

	/**
	 * Instantiates a new classifier selection. By default considers all
	 * classifiers.
	 */
	public ClassifierSelection() {
		this(Sets.newHashSet(Classifiers.values()));
	}

	/**
	 * Instantiates a new classifier selection.
	 * 
	 * @param candidateClassifiers
	 *            the set of {@link Classifiers} to consider
	 * @param folds
	 *            the number of folds of cross validation to perform
	 */
	public ClassifierSelection(Set<Classifiers> candidateClassifiers, int folds) {
		this.candidateClassifiers = candidateClassifiers;
		this.folds = folds;
		evaluator = new FoldEvaluator(folds);
	}

	/**
	 * Instantiates a new classifier selection. By default considers all
	 * classifiers
	 * 
	 * @param folds
	 *            the number of folds of cross validation to perform
	 */
	public ClassifierSelection(int folds) {
		this(Sets.newHashSet(Classifiers.values()), folds);
	}

	/**
	 * returns a classifier optimized on the specified objective function using
	 * cross validation on the input data. uses an additional "bias" term in
	 * models, (models don't necessarily pass through origin)
	 * 
	 * @param inputData
	 *            data used for training and evaluation
	 * @param score
	 *            used to measure model quality
	 * @return model ready to be trained
	 */
	Classifier<?> optimizedClassifier(BinaryClassificationInstances inputData,
			ObjectiveScorer<BinaryClassificationTarget> score) {
		return optimizedClassifier(inputData, score, true);
	}

	/**
	 * returns a classifier optimized on the specified objective function using
	 * cross validation on the input data.
	 * 
	 * @param inputData
	 *            data used for training and evaluation
	 * @param score
	 *            used to measure model quality
	 * @param bias
	 *            should the model pass through the origin?
	 * @return model ready to be trained
	 */
	Classifier<?> optimizedClassifier(BinaryClassificationInstances inputData,
			ObjectiveScorer<BinaryClassificationTarget> score, boolean bias) {
		return this.optimize(inputData, score, bias).getClassifier(
				inputData.getDimensions(), bias);
	}

	/**
	 * returns the Classifiers enum representing the model with the best
	 * performance on the input data according to the specified objective the
	 * models have considered an additional bias term to avoid passing through
	 * the origin. the classifiers object can be used to generate trainable
	 * models
	 * 
	 * @param inputData
	 *            training data used to find the best classifier through
	 *            x-validation
	 * @param scorer
	 *            used to measure model quality
	 * @return model ready to be trained
	 */
	Classifiers optimize(BinaryClassificationInstances inputData,
			ObjectiveScorer<BinaryClassificationTarget> scorer) {
		return optimize(inputData, scorer, true);
	}

	/**
	 * returns the Classifiers enum representing the model with the best
	 * performance on the input data according to the specified objective allow
	 * the user to specify a bias term allowing the model to not pass through
	 * the origin. the classifiers object can be used to generate trainable
	 * models
	 * 
	 * @param inputData
	 *            training data used to find the best classifier through
	 *            x-validation
	 * @param scorer
	 *            used to measure model quality
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return model ready to be trained
	 */
	Classifiers optimize(BinaryClassificationInstances inputData,
			ObjectiveScorer<BinaryClassificationTarget> scorer, boolean bias) {
		Classifiers winner = null;
		double objectiveValue = Double.MIN_VALUE;

		for (Classifiers classifiers : candidateClassifiers) {
			ClassifierBuilder<?, ?> builder = classifiers.getClassifierBuilder(
					inputData.getDimensions(), bias);

			double perf = evaluator.evaluate(inputData, scorer, builder);

			LOGGER.info(classifiers + " has an objective value of: " + perf);
			if (perf > objectiveValue) {
				objectiveValue = perf;
				winner = classifiers;
			}
		}

		return winner;
	}

}
