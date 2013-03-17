package com.parallax.ml.classifier;

import com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions;
import com.parallax.ml.classifier.bayes.NaiveBayesBuilder;
import com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.BudgetKernelPerceptronBuilder;
import com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.ForgetronBuilder;
import com.parallax.ml.classifier.lazy.SequentialKNNBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.AROWClassifierBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PassiveAggressiveBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PegasosBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PerceptronWithMarginBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.WinnowClassifierBuilder;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.OptionSet;

/**
 * A factory for building {@link ClassifierBuilder}'s from their associated
 * configs Also gives a list of builder types.
 * 
 * Note, {@link Classifiers} should probably be preferred.
 */
public class ClassifierConfigurationFactory {

	/**
	 * returns a classifier builder from an arbitrary Configuration built on a
	 * sub class of ClassifierBuilder
	 * 
	 * this is hideous fuck my life.
	 * 
	 * @param <B>
	 *            the type of builder desired
	 * @param conf
	 *            configuration for the desired classifier builder
	 * @return the builder for a particular classifier.
	 */
	@SuppressWarnings("unchecked")
	public static <B extends ClassifierBuilder<?, ?>> B buildClassifierBuilder(
			Configuration<? extends ClassifierBuilder<?, ?>> conf) {
		OptionSet<?> options = conf.getOptionSet();
		@SuppressWarnings("rawtypes")
		Classifiers cType = ((ClassifierOptions) options).getClassifierType();
		return (B) cType.getClassifierBuilder(conf);
	}

	/**
	 * Builds the winnow classifier builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the winnow classifier builder
	 */
	public static WinnowClassifierBuilder buildWinnowClassifierBuilder(
			int dimension, boolean bias) {
		return new WinnowClassifierBuilder(dimension, bias);
	}

	/**
	 * Builds the winnow classifier builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the winnow classifier builder
	 */
	public static WinnowClassifierBuilder buildWinnowClassifierBuilder(
			Configuration<WinnowClassifierBuilder> config) {
		return new WinnowClassifierBuilder(config);
	}

	/**
	 * Builds the winnow classifier builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<WinnowClassifierBuilder> buildWinnowClassifierBuilderConfiguration() {
		return new Configuration<WinnowClassifierBuilder>(
				WinnowClassifierBuilder.options);
	}

	/**
	 * Builds the perceptron with margin builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the perceptron with margin builder
	 */
	public static PerceptronWithMarginBuilder buildPerceptronWithMarginBuilder(
			int dimension, boolean bias) {
		return new PerceptronWithMarginBuilder(dimension, bias);
	}

	/**
	 * Builds the perceptron with margin builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the perceptron with margin builder
	 */
	public static PerceptronWithMarginBuilder buildPerceptronWithMarginBuilder(
			Configuration<PerceptronWithMarginBuilder> config) {
		return new PerceptronWithMarginBuilder(config);
	}

	/**
	 * Builds the perceptron with margin builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<PerceptronWithMarginBuilder> buildPerceptronWithMarginBuilderConfiguration() {
		return new Configuration<PerceptronWithMarginBuilder>(
				PerceptronWithMarginBuilder.options);
	}

	/**
	 * Builds the passive aggressive builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the passive aggressive builder
	 */
	public static PassiveAggressiveBuilder buildPassiveAggressiveBuilder(
			int dimension, boolean bias) {
		return new PassiveAggressiveBuilder(dimension, bias);
	}

	/**
	 * Builds the passive aggressive builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the passive aggressive builder
	 */
	public static PassiveAggressiveBuilder buildPassiveAggressiveBuilder(
			Configuration<PassiveAggressiveBuilder> config) {
		return new PassiveAggressiveBuilder(config);
	}

	/**
	 * Builds the passive aggressive builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<PassiveAggressiveBuilder> buildPassiveAggressiveBuilderConfiguration() {
		return new Configuration<PassiveAggressiveBuilder>(
				PassiveAggressiveBuilder.options);
	}

	/**
	 * Builds the pegasos builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the pegasos builder
	 */
	public static PegasosBuilder buildPegasosBuilder(int dimension, boolean bias) {
		return new PegasosBuilder(dimension, bias);
	}

	/**
	 * Builds the pegasos builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the pegasos builder
	 */
	public static PegasosBuilder buildPegasosBuilder(
			Configuration<PegasosBuilder> config) {
		return new PegasosBuilder(config);
	}

	/**
	 * Builds the pegasos builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<PegasosBuilder> buildPegasosBuilderConfiguration() {
		return new Configuration<PegasosBuilder>(PegasosBuilder.options);
	}

	/**
	 * Builds the logistic regression builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the logistic regression builder
	 */
	public static LogisticRegressionBuilder buildLogisticRegressionBuilder(
			int dimension, boolean bias) {
		return new LogisticRegressionBuilder(dimension, bias);
	}

	/**
	 * Builds the logistic regression builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the logistic regression builder
	 */
	public static LogisticRegressionBuilder buildLogisticRegressionBuilder(
			Configuration<LogisticRegressionBuilder> config) {
		return new LogisticRegressionBuilder(config);
	}

	/**
	 * Builds the logistic regression builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<LogisticRegressionBuilder> buildLogisticRegressionBuilderConfiguration() {
		return new Configuration<LogisticRegressionBuilder>(
				LogisticRegressionBuilder.options);
	}

	/**
	 * Builds the arow classifier builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the aROW classifier builder
	 */
	public static AROWClassifierBuilder buildAROWClassifierBuilder(
			int dimension, boolean bias) {
		return new AROWClassifierBuilder(dimension, bias);
	}

	/**
	 * Builds the arow classifier builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the aROW classifier builder
	 */
	public static AROWClassifierBuilder buildAROWClassifierBuilder(
			Configuration<AROWClassifierBuilder> config) {
		return new AROWClassifierBuilder(config);
	}

	/**
	 * Builds the arow classifier builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<AROWClassifierBuilder> buildAROWClassifierBuilderConfiguration() {
		return new Configuration<AROWClassifierBuilder>(
				AROWClassifierBuilder.options);
	}

	/**
	 * Builds the naive bayes builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the naive bayes builder
	 */
	public static NaiveBayesBuilder buildNaiveBayesBuilder(int dimension,
			boolean bias) {
		return new NaiveBayesBuilder(dimension, bias);
	}

	/**
	 * Builds the naive bayes builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the naive bayes builder
	 */
	public static NaiveBayesBuilder buildNaiveBayesBuilder(
			Configuration<NaiveBayesBuilder> config) {
		return new NaiveBayesBuilder(config);
	}

	/**
	 * Builds the naive bayes builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<NaiveBayesBuilder> buildNaiveBayesBuilderConfiguration() {
		return new Configuration<NaiveBayesBuilder>(NaiveBayesBuilder.options);
	}

	/**
	 * Builds the sequential knn builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the sequential knn builder
	 */
	public static SequentialKNNBuilder buildSequentialKNNBuilder(int dimension,
			boolean bias) {
		return new SequentialKNNBuilder(dimension, bias);
	}

	/**
	 * Builds the sequential knn builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the sequential knn builder
	 */
	public static SequentialKNNBuilder buildSequentialKNNBuilder(
			Configuration<SequentialKNNBuilder> config) {
		return new SequentialKNNBuilder(config);
	}

	/**
	 * Builds the sequential knn builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<SequentialKNNBuilder> buildSequentialKNNBuilderConfiguration() {
		return new Configuration<SequentialKNNBuilder>(
				SequentialKNNBuilder.options);
	}

	/**
	 * Builds the forgetron builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the forgetron builder
	 */
	public static ForgetronBuilder buildForgetronBuilder(int dimension,
			boolean bias) {
		return new ForgetronBuilder(dimension, bias);
	}

	/**
	 * Builds the forgetron builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the forgetron builder
	 */
	public static ForgetronBuilder buildForgetronBuilder(
			Configuration<ForgetronBuilder> config) {
		return new ForgetronBuilder(config);
	}

	/**
	 * Builds the forgetron builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<ForgetronBuilder> buildForgetronBuilderConfiguration() {
		return new Configuration<ForgetronBuilder>(ForgetronBuilder.options);
	}

	/**
	 * Builds the budget kernel perceptron builder.
	 * 
	 * @param dimension
	 *            dimension the number of features in the instantiated
	 *            classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the budget kernel perceptron builder
	 */
	public static BudgetKernelPerceptronBuilder buildBudgetKernelPerceptronBuilder(
			int dimension, boolean bias) {
		return new BudgetKernelPerceptronBuilder(dimension, bias);
	}

	/**
	 * Builds the budget kernel perceptron builder.
	 * 
	 * @param config
	 *            the configuration containing the desired model settings
	 * @return the budget kernel perceptron builder
	 */
	public static BudgetKernelPerceptronBuilder buildBudgetKernelPerceptronBuilder(
			Configuration<BudgetKernelPerceptronBuilder> config) {
		return new BudgetKernelPerceptronBuilder(config);
	}

	/**
	 * Builds the budget kernel perceptron builder configuration.
	 * 
	 * @return a configuration for the desired classifier type
	 */
	public static Configuration<BudgetKernelPerceptronBuilder> buildBudgetKernelPerceptronBuilderConfiguration() {
		return new Configuration<BudgetKernelPerceptronBuilder>(
				BudgetKernelPerceptronBuilder.options);
	}
}
