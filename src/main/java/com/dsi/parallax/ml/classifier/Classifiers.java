/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier;

import com.dsi.parallax.ml.classifier.bayes.NaiveBayes;
import com.dsi.parallax.ml.classifier.bayes.NaiveBayesBuilder;
import com.dsi.parallax.ml.classifier.kernelmethods.BudgetKernelPerceptron;
import com.dsi.parallax.ml.classifier.kernelmethods.Forgetron;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.BudgetKernelPerceptronBuilder;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.ForgetronBuilder;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.KernelSGDBuilder;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelSGD;
import com.dsi.parallax.ml.classifier.lazy.LocalLogisticRegression;
import com.dsi.parallax.ml.classifier.lazy.LocalLogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.lazy.SequentialKNN;
import com.dsi.parallax.ml.classifier.lazy.SequentialKNNBuilder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradienUpdateableQuadraticSVM;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableL2Builder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableLogisticRegressionBuilder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableModifiedHuberBuilder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableQuadraticSVMBuilder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableL2;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableLogisticRegression;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableModifiedHuber;
import com.dsi.parallax.ml.classifier.linear.updateable.*;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.*;
import com.dsi.parallax.ml.classifier.rules.MeanClassifier;
import com.dsi.parallax.ml.classifier.rules.MeanClassifier.MeanClassifierBuilder;
import com.dsi.parallax.ml.classifier.rules.ModeClassifier;
import com.dsi.parallax.ml.classifier.rules.ModeClassifier.ModeClassifierBuilder;
import com.dsi.parallax.ml.classifier.trees.ID3Builder;
import com.dsi.parallax.ml.classifier.trees.ID3TreeClassifier;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.OptionSet;
import com.google.common.collect.Maps;

import java.util.Map;

/**
 * an enumeration on the type of classifiers. also defines some utility methods
 * for buiding classifiers, configurations, and builders.
 * 
 * @author josh
 * 
 */
public enum Classifiers {

	/** {@link AROWClassifier} */
	AROW {
		@Override
		public Class<?> getClassifierClass() {
			return AROWClassifier.class;
		}

		@Override
		public String getName() {
			return "arow";
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<AROWClassifierBuilder> builderConf = (Configuration<AROWClassifierBuilder>) conf;
			return new AROWClassifierBuilder(builderConf);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new AROWClassifierBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return AROWClassifierBuilder.getOptions();
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new AROWClassifier(dimension, bias);
		}

	},

	/** {@link WinnowClassifier} */
	WINNOW {
		@Override
		public Class<?> getClassifierClass() {
			return WinnowClassifier.class;
		}

		@Override
		public String getName() {
			return "winnow";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new WinnowClassifier(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<WinnowClassifierBuilder> builderConf = (Configuration<WinnowClassifierBuilder>) conf;
			return new WinnowClassifierBuilder(builderConf);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new WinnowClassifierBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return WinnowClassifierBuilder.getOptions();
		}
	},

	/** {@link Pegasos} */
	PEGASOS {
		@Override
		public Class<?> getClassifierClass() {
			return Pegasos.class;
		}

		@Override
		public String getName() {
			return "pegasos";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new Pegasos(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<PegasosBuilder> builderConf = (Configuration<PegasosBuilder>) conf;
			return new PegasosBuilder(builderConf);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new PegasosBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return PegasosBuilder.getOptions();
		}

	},

	/** {@link PassiveAggressive} */
	PASSIVEAGGRESSIVE {
		@Override
		public Class<?> getClassifierClass() {
			return PassiveAggressive.class;
		}

		@Override
		public String getName() {
			return "pa";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new PassiveAggressive(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<PassiveAggressiveBuilder> builderConf = (Configuration<PassiveAggressiveBuilder>) conf;
			return new PassiveAggressiveBuilder(builderConf);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new PassiveAggressiveBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return PassiveAggressiveBuilder.getOptions();
		}
	},

	/** {@link NaiveBayes} */
	NB {
		@Override
		public Class<?> getClassifierClass() {
			return NaiveBayes.class;
		}

		@Override
		public String getName() {
			return "nb";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new NaiveBayes(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<NaiveBayesBuilder> builderConfig = (Configuration<NaiveBayesBuilder>) conf;
			return new NaiveBayesBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new NaiveBayesBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return NaiveBayesBuilder.getOptions();
		}
	},

	/** {@link Forgetron} */
	FORGETRON {
		@Override
		public Class<?> getClassifierClass() {
			return Forgetron.class;
		}

		@Override
		public String getName() {
			return "forgetron";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new Forgetron(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<ForgetronBuilder> builderConfig = (Configuration<ForgetronBuilder>) conf;
			return new ForgetronBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new ForgetronBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return ForgetronBuilder.getOptions();
		}
	},

	/** {@link LogisticRegression} */
	LOGISTICREGRESSION {
		@Override
		public Class<?> getClassifierClass() {
			return LogisticRegression.class;
		}

		@Override
		public String getName() {
			return "lr";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new LogisticRegression(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<LogisticRegressionBuilder> builderConfig = (Configuration<LogisticRegressionBuilder>) conf;
			return new LogisticRegressionBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new LogisticRegressionBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return LogisticRegressionBuilder.getOptions();
		}

	},

	/** {@link BudgetKernelPerceptron} */
	BUDGETKERNELPERCEPTRON {
		@Override
		public Class<?> getClassifierClass() {
			return BudgetKernelPerceptron.class;
		}

		@Override
		public String getName() {
			return "bkp";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new BudgetKernelPerceptron(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<BudgetKernelPerceptronBuilder> builderConfig = (Configuration<BudgetKernelPerceptronBuilder>) conf;
			return new BudgetKernelPerceptronBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new BudgetKernelPerceptronBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return BudgetKernelPerceptronBuilder.getOptions();
		}
	},

	/** {@link PerceptronWithMargin}. */
	MARGINPERCEPTRON {
		@Override
		public Class<?> getClassifierClass() {
			return PerceptronWithMargin.class;
		}

		@Override
		public String getName() {
			return "mp";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new PerceptronWithMargin(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<PerceptronWithMarginBuilder> builderConfig = (Configuration<PerceptronWithMarginBuilder>) conf;
			return new PerceptronWithMarginBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new PerceptronWithMarginBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return PerceptronWithMarginBuilder.getOptions();
		}
	},

	/** {@link KernelSGD} */
	KERNELSGD {
		@Override
		public Class<?> getClassifierClass() {
			return KernelSGD.class;
		}

		@Override
		public String getName() {
			return "ksgd";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new KernelSGD(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<KernelSGDBuilder> builderConfig = (Configuration<KernelSGDBuilder>) conf;
			return new KernelSGDBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new KernelSGDBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return KernelSGDBuilder.getOptions();
		}
	},

	/** {@link SequentialKNN} */
	KNN {
		@Override
		public Class<?> getClassifierClass() {
			return SequentialKNN.class;
		}

		@Override
		public String getName() {
			return "knn";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new SequentialKNN(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<SequentialKNNBuilder> builderConfig = (Configuration<SequentialKNNBuilder>) conf;
			return new SequentialKNNBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new SequentialKNNBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return SequentialKNNBuilder.getOptions();
		}
	},

	/** {@link ModeClassifier} */
	MODE {

		@Override
		public Class<?> getClassifierClass() {
			return ModeClassifier.class;
		}

		@Override
		public String getName() {
			return "mode";
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new ModeClassifier(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<ModeClassifierBuilder> builderConfig = (Configuration<ModeClassifierBuilder>) conf;
			return new ModeClassifierBuilder(builderConfig);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new ModeClassifierBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return ModeClassifierBuilder.getOptions();
		}
	},

	/** {@link ModeClassifier} */
	MEAN {

		@Override
		public String getName() {
			return "mean";
		}

		@Override
		public Class<?> getClassifierClass() {
			return MeanClassifier.class;
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<MeanClassifierBuilder> builderConfig = (Configuration<MeanClassifierBuilder>) conf;
			return new MeanClassifierBuilder(builderConfig);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new MeanClassifierBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return MeanClassifierBuilder.getOptions();
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new MeanClassifier(dimension, bias);
		}
	},

	/** {@link ID3TreeClassifier} */
	ID3 {

		@Override
		public String getName() {
			return "id3";
		}

		@Override
		public Class<?> getClassifierClass() {
			return ID3TreeClassifier.class;
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new ID3TreeClassifier(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<ID3Builder> builderConfig = (Configuration<ID3Builder>) conf;
			return new ID3Builder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new ID3Builder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			// TODO: because of leaf builders, this is complex at the moment.
			throw new UnsupportedOperationException(
					"get options isnt supported for ID3 at the moment");
		}

	},

	/** {@link GradientUpdateableL2} */
	L2 {

		@Override
		public String getName() {
			return "l2";
		}

		@Override
		public Class<GradientUpdateableL2> getClassifierClass() {
			return GradientUpdateableL2.class;
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new GradientUpdateableL2(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<GradientUpdateableL2Builder> builderConfig = (Configuration<GradientUpdateableL2Builder>) conf;
			return new GradientUpdateableL2Builder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new GradientUpdateableL2Builder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return GradientUpdateableL2Builder.getOptions();
		}

	},

	/** {@link GradientOptimizableQuadraticSVM} */
	QSVM {

		@Override
		public String getName() {
			return "qsvm";
		}

		@Override
		public Class<?> getClassifierClass() {
			return GradienUpdateableQuadraticSVM.class;
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new GradienUpdateableQuadraticSVM(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<GradientUpdateableQuadraticSVMBuilder> builderConfig = (Configuration<GradientUpdateableQuadraticSVMBuilder>) conf;
			return new GradientUpdateableQuadraticSVMBuilder(builderConfig);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new GradientUpdateableQuadraticSVMBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return GradientUpdateableQuadraticSVMBuilder.getOptions();
		}

	},

	/** {@link GradientUpdateableModifiedHuber} */
	HUBER {

		@Override
		public String getName() {
			return "huber";
		}

		@Override
		public Class<?> getClassifierClass() {
			return GradientUpdateableModifiedHuber.class;
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new GradientUpdateableModifiedHuber(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<GradientUpdateableModifiedHuberBuilder> builderConfig = (Configuration<GradientUpdateableModifiedHuberBuilder>) conf;
			return new GradientUpdateableModifiedHuberBuilder(builderConfig);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new GradientUpdateableModifiedHuberBuilder(dimension, bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return GradientUpdateableModifiedHuberBuilder.getOptions();
		}

	},
	/** {@link GradientUpdateableLogisticRegression} */
	OPTLR {

		@Override
		public String getName() {
			return "optlr";
		}

		@Override
		public Class<?> getClassifierClass() {
			return GradientUpdateableLogisticRegression.class;
		}

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new GradientUpdateableLogisticRegression(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<GradientUpdateableLogisticRegressionBuilder> builderConfig = (Configuration<GradientUpdateableLogisticRegressionBuilder>) conf;
			return new GradientUpdateableLogisticRegressionBuilder(
					builderConfig);

		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new GradientUpdateableLogisticRegressionBuilder(dimension,
					bias);
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return GradientUpdateableLogisticRegressionBuilder.getOptions();
		}

	},
	LOCALLR {

		@Override
		public Classifier<?> getClassifier(int dimension, boolean bias) {
			return new LocalLogisticRegression(dimension, bias);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(
				Configuration<? extends ClassifierBuilder<?, ?>> conf) {
			@SuppressWarnings("unchecked")
			Configuration<LocalLogisticRegressionBuilder> builderConfig = (Configuration<LocalLogisticRegressionBuilder>) conf;
			return new LocalLogisticRegressionBuilder(builderConfig);
		}

		@Override
		public ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
				boolean bias) {
			return new LocalLogisticRegressionBuilder(dimension, bias);
		}

		@Override
		public String getName() {
			return "locallr";
		}

		@Override
		public Class<?> getClassifierClass() {
			return LocalLogisticRegression.class;
		}

		@Override
		public OptionSet<? extends ClassifierBuilder<?, ?>> getOptions() {
			return LocalLogisticRegressionBuilder.getOptions();
		}

	};

	/**
	 * factory for the associated classifier type.
	 * 
	 * @param dimension
	 *            input dimension of the model
	 * @param bias
	 *            should the model include an extra intercept term?
	 * @return an instantiated classifier
	 */
	public abstract Classifier<?> getClassifier(int dimension, boolean bias);

	/**
	 * factory for the associated classifier type.
	 * 
	 * @param conf
	 *            configuration object defining the model's settings
	 * @return an instantiated classifier
	 */
	public abstract ClassifierBuilder<?, ?> getClassifierBuilder(
			Configuration<? extends ClassifierBuilder<?, ?>> conf);

	/**
	 * Gets the classifier builder.
	 * 
	 * @param dimension
	 *            input dimension of the model
	 * @param bias
	 *            should the model include an extra intercept term?
	 * @return the classifier builder
	 */
	public abstract ClassifierBuilder<?, ?> getClassifierBuilder(int dimension,
			boolean bias);

	/**
	 * get a string alias of the classifier type. example, LogisticRegression ->
	 * lr
	 * 
	 * @return string alias
	 */
	public abstract String getName();

	/**
	 * get the class of the assoicated classifier type.
	 * 
	 * @return model's class
	 */
	public abstract Class<?> getClassifierClass();

	/**
	 * get the set of options associated with a classifier type.
	 * 
	 * @return OptionSet assoicated with classifier type
	 */
	public abstract OptionSet<? extends ClassifierBuilder<?, ?>> getOptions();

	/**
	 * Gets the configuration for a builder of the desired classifier type
	 * 
	 * @return the configuration for the associated classifier builder
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Configuration<?> getConfiguration() {
		OptionSet<?> options = this.getOptions();
		return new Configuration(options);
	}

	/** A mapping of classifier "names" to classifier types */
	private static Map<String, Classifiers> nameToClassifierType = Maps
			.newHashMap();

	static {
		for (Classifiers type : Classifiers.values())
			nameToClassifierType.put(type.getName(), type);
	}

	/** Mapping of Classifier Class to Classifier types */
	private static Map<Class<?>, Classifiers> classToClassifierType = Maps
			.newHashMap();

	static {
		for (Classifiers type : Classifiers.values())
			classToClassifierType.put(type.getClassifierClass(), type);
	}

	/**
	 * Build a classifier of the desired type.
	 * 
	 * @param name
	 *            the name of the desired classifier
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the classifier
	 */
	public static Classifier<?> getClassifier(String name, int dimension,
			boolean bias) {
		if (!nameToClassifierType.containsKey(name))
			throw new IllegalArgumentException(name
					+ " is not a valid model name. Options are: "
					+ nameToClassifierType.keySet());
		return nameToClassifierType.get(name).getClassifier(dimension, bias);
	}

	/**
	 * Builds a classifier of the desired class
	 * 
	 * @param clazz
	 *            the class of the desired classifier
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 * @return the classifier
	 */
	public static Classifier<?> getClassifier(Class<?> clazz, int dimension,
			boolean bias) {
		if (!classToClassifierType.containsKey(clazz))
			throw new IllegalArgumentException(clazz
					+ " is not a valid model class. Options are: "
					+ classToClassifierType.keySet());
		return classToClassifierType.get(clazz).getClassifier(dimension, bias);
	}

	/**
	 * Build options for a builder of the desired classifier type
	 * 
	 * @param name
	 *            the name of the desired classifier
	 * @return the options
	 */
	public static OptionSet<?> getOptions(String name) {
		if (!nameToClassifierType.containsKey(name))
			throw new IllegalArgumentException(name
					+ " is not a valid model name. Options are: "
					+ nameToClassifierType.keySet());
		return nameToClassifierType.get(name).getOptions();
	}

	/**
	 * Build options for the builder of the desired classifier type
	 * 
	 * @param clazz
	 *            the Class of the desired classifier
	 * @return the options
	 */
	public static OptionSet<?> getOptions(Class<?> clazz) {
		if (!classToClassifierType.containsKey(clazz))
			throw new IllegalArgumentException(clazz
					+ " is not a valid model class. Options are: "
					+ classToClassifierType.keySet());
		return classToClassifierType.get(clazz).getOptions();
	}

	/**
	 * Gets the classifier Class based on the classifier's name
	 * 
	 * @param name
	 *            the name of the desired classifier
	 * @return the classifier class
	 */
	public static Class<?> getClassifierClass(String name) {
		if (!nameToClassifierType.containsKey(name))
			throw new IllegalArgumentException(name
					+ " is not a valid model name. Options are: "
					+ nameToClassifierType.keySet());
		return nameToClassifierType.get(name).getClass();
	}

	/**
	 * Gets the name of the desired classifier based on it's Class
	 * 
	 * @param clazz
	 *            the Class of the desired classifier
	 * @return the classifier name
	 */
	public static String getClassifierName(Class<?> clazz) {
		if (!classToClassifierType.containsKey(clazz))
			throw new IllegalArgumentException(clazz
					+ " is not a valid model class. Options are: "
					+ classToClassifierType.keySet());
		return classToClassifierType.get(clazz).getName();
	}

	/**
	 * Gets the classifier type ({@link Classifiers}) based on the classifier
	 * Name
	 * 
	 * @param name
	 *            the name of the desired classifier
	 * @return the classifier type
	 */
	public static Classifiers getClassifierType(String name) {
		if (!nameToClassifierType.containsKey(name))
			throw new IllegalArgumentException(name
					+ " is not a valid model name. Options are: "
					+ nameToClassifierType.keySet());
		return nameToClassifierType.get(name);
	}

	/**
	 * Gets the classifier type ({@link Classifiers}) based on the classifier's
	 * Class.
	 * 
	 * @param clazz
	 *            the Class of the desired classifier
	 * @return the classifier type
	 */
	public static Classifiers getClassifierType(Class<?> clazz) {
		if (!classToClassifierType.containsKey(clazz))
			throw new IllegalArgumentException(clazz
					+ " is not a valid model class. Options are: "
					+ classToClassifierType.keySet());
		return classToClassifierType.get(clazz);
	}

}
