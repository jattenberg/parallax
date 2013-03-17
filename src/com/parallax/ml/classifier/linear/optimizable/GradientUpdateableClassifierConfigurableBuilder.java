package com.parallax.ml.classifier.linear.optimizable;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.UpdateableClassifierBuilder;
import com.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.parallax.ml.util.bounds.GreaterThanValueBound;
import com.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.parallax.ml.util.option.ConfigurableOption;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.IntegerOption;
import com.parallax.ml.util.option.OptionSet;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.stochastic.SGDBuilder;
import com.parallax.optimization.stochastic.StochasticGradientOptimizationBuilder;

// TODO: Auto-generated Javadoc
/**
 * The Class GradientUpdateableClassifierConfigurableBuilder.
 * 
 * @param <C>
 *            the generic type
 * @param <B>
 *            the generic type
 */
public abstract class GradientUpdateableClassifierConfigurableBuilder<C extends GradientUpdateableClassifier<C>, B extends GradientUpdateableClassifierConfigurableBuilder<C, B>>
		extends UpdateableClassifierBuilder<C, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -2085396149352796591L;

	/** The passes. */
	protected int passes = 3;

	/** The mini batch size. */
	protected int miniBatchSize = 5;

	/** The builder. */
	protected StochasticGradientOptimizationBuilder<?> builder = null;

	/** The truncation builder. */
	protected TruncationConfigurableBuilder truncationBuilder = new TruncationConfigurableBuilder();

	/**
	 * Instantiates a new gradient updateable classifier configurable builder.
	 * 
	 * @param config
	 *            the config
	 */
	protected GradientUpdateableClassifierConfigurableBuilder(
			Configuration<B> config) {
		super(config);
	}

	/**
	 * Instantiates a new gradient updateable classifier configurable builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	protected GradientUpdateableClassifierConfigurableBuilder(int dimension,
			boolean bias) {
		super(dimension, bias);
	}

	/**
	 * instantiates a new
	 * {@link GradientUpdateableClassifierConfigurableBuilder} dimension and
	 * bias must be set manually using setters
	 */
	public GradientUpdateableClassifierConfigurableBuilder() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#setPasses(int)
	 */
	public B setPasses(int passes) {
		checkArgument(passes > 0, "passes must be positive, given: %s", passes);
		this.passes = passes;
		return thisBuilder;
	}

	/**
	 * Sets the mini batch size.
	 * 
	 * @param miniBatchSize
	 *            the mini batch size
	 * @return the b
	 */
	public B setMiniBatchSize(int miniBatchSize) {
		checkArgument(miniBatchSize >= 1,
				"minibatch size must be positive, given: %s", miniBatchSize);
		this.miniBatchSize = miniBatchSize;
		return thisBuilder;
	}

	/**
	 * Sets the truncation builder.
	 * 
	 * @param truncationBuilder
	 *            the truncation builder
	 * @return the b
	 */
	public B setTruncationBuilder(
			TruncationConfigurableBuilder truncationBuilder) {
		this.truncationBuilder = truncationBuilder;
		return thisBuilder;
	}

	/**
	 * Sets the truncation builder.
	 * 
	 * @param truncationConfiguration
	 *            the truncation configuration
	 * @return the b
	 */
	public B setTruncationBuilder(
			Configuration<TruncationConfigurableBuilder> truncationConfiguration) {
		return setTruncationBuilder(new TruncationConfigurableBuilder(
				truncationConfiguration));
	}

	public <O extends StochasticGradientOptimizationBuilder<O>> B setOptimizationBuilder(
			O optimizer) {
		checkArgument(
				optimizer.getDimension() == getDimension(),
				"optimization builder dimension (%s) should match model dimension (%s)",
				optimizer.getDimension(), getDimension());
		checkArgument(optimizer.getBias() == this.bias,
				"optimization builder bias (%s) should match model bias (%s)",
				optimizer.getBias(), bias);
		this.builder = optimizer;
		return thisBuilder;
	}

	public StochasticGradientOptimizationBuilder<?> getOptimizationBuilder() {
		return this.builder;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#getConfiguration()
	 */
	@Override
	public Configuration<B> getConfiguration() {
		Configuration<B> conf = super.getConfiguration();
		return populateConfiguration(conf);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#populateConfiguration
	 * (com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<B> populateConfiguration(Configuration<B> conf) {
		super.populateConfiguration(conf);
		conf.addConfigurableValueOnShortName("T",
				truncationBuilder.getConfiguration());
		conf.addIntegerValueOnShortName("p", passes);
		conf.addIntegerValueOnShortName("m", miniBatchSize);
		return conf;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#configure(com.
	 * parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<B> conf) {
		super.configure(conf);
		@SuppressWarnings("unchecked")
		Configuration<TruncationConfigurableBuilder> truncConfig = (Configuration<TruncationConfigurableBuilder>) conf
				.configurationFromShortName("T");
		setTruncationBuilder(truncConfig);
		setPasses(conf.integerOptionFromShortName("p"));
		setMiniBatchSize(conf.integerOptionFromShortName("m"));
	}

	protected StochasticGradientOptimizationBuilder<?> initializeGradientBuilder() {
		if (builder == null) {
			builder = new SGDBuilder(getDimension(), bias);
		}
		return builder;
	}

	/**
	 * The Class GradientUpdateableOptions.
	 * 
	 * @param <C>
	 *            the generic type
	 * @param <B>
	 *            the generic type
	 */
	protected abstract static class GradientUpdateableOptions<C extends AbstractGradientUpdateableClassifier<C>, B extends GradientUpdateableClassifierConfigurableBuilder<C, B>>
			extends UpdateableOptions<C, B> {
		{
			addOption(new IntegerOption("p", "passes",
					"number of passes used for batch mode training", 3, false,
					new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(10000)));
			addOption(new IntegerOption("m", "minibatches",
					"minibatch size when estimating gradients", 5, false,
					new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new ConfigurableOption<TruncationConfigurableBuilder>(
					"T", "truncationConfig", false,
					"the configuration for builders of gradient truncations. Options: "
							+ TruncationConfigurableBuilder.optionInfo(),
					new Configuration<TruncationConfigurableBuilder>(
							TruncationConfigurableBuilder.options)));
		}
	}

	/**
	 * The Class GradientUpdateableL2Builder.
	 */
	public static class GradientUpdateableL2Builder
			extends
			GradientUpdateableClassifierConfigurableBuilder<GradientUpdateableL2, GradientUpdateableL2Builder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 6818228602046314281L;

		/** The options. */
		public static OptionSet<GradientUpdateableL2Builder> options = new GradientUpdateableL2Options();

		/**
		 * Instantiates a new gradient updateable l2 builder.
		 * 
		 * @param config
		 *            the config
		 */
		public GradientUpdateableL2Builder(
				Configuration<GradientUpdateableL2Builder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new gradient updateable l2 builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public GradientUpdateableL2Builder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new {@link GradientUpdateableL2Builder} dimension and
		 * bias must be set manually using setters
		 */
		public GradientUpdateableL2Builder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public GradientUpdateableL2 build() {

			GradientUpdateableL2 model = new GradientUpdateableL2(
					initializeGradientBuilder().setGradientTruncationBuilder(
							truncationBuilder), getDimension(), bias);
			model.setPasses(passes)
					.setSmoothertype(regType)
					.setCrossvalidateSmootherTraining(
							crossValidateSmootherTraining).initialize();
			return model;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected GradientUpdateableL2Builder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<GradientUpdateableL2Builder> conf) {
			super.configure(conf);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder#getConfiguration()
		 */
		@Override
		public Configuration<GradientUpdateableL2Builder> getConfiguration() {
			Configuration<GradientUpdateableL2Builder> conf = new Configuration<GradientUpdateableL2Builder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<GradientUpdateableL2Builder> populateConfiguration(
				Configuration<GradientUpdateableL2Builder> conf) {
			super.populateConfiguration(conf);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static GradientUpdateableL2Options getOptions() {
			return new GradientUpdateableL2Options();
		}

		/**
		 * The Class GradientUpdateableL2Options.
		 */
		protected static class GradientUpdateableL2Options
				extends
				GradientUpdateableOptions<GradientUpdateableL2, GradientUpdateableL2Builder> {

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.L2;
			}

		}
	}

	/**
	 * The Class GradientUpdateableQuadraticSVMBuilder.
	 */
	public static class GradientUpdateableQuadraticSVMBuilder
			extends
			GradientUpdateableClassifierConfigurableBuilder<GradienUpdateableQuadraticSVM, GradientUpdateableQuadraticSVMBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 6818228602046314281L;

		/** The gamma. */
		private double gamma = 2.0;

		/** The options. */
		public static OptionSet<GradientUpdateableQuadraticSVMBuilder> options = new GradientUpdateableQuadraticSVMOptions();

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param config
		 *            the config
		 */
		public GradientUpdateableQuadraticSVMBuilder(
				Configuration<GradientUpdateableQuadraticSVMBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public GradientUpdateableQuadraticSVMBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new {@link GradientUpdateableQuadraticSVMBuilder}
		 * dimension and bias must be set manually using setters
		 */
		public GradientUpdateableQuadraticSVMBuilder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public GradienUpdateableQuadraticSVM build() {
			GradienUpdateableQuadraticSVM model = new GradienUpdateableQuadraticSVM(
					initializeGradientBuilder().setGradientTruncationBuilder(
							truncationBuilder), getDimension(), bias);
			model.setGamma(gamma)
					.setPasses(passes)
					.setSmoothertype(regType)
					.setCrossvalidateSmootherTraining(
							crossValidateSmootherTraining).initialize();
			return model;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected GradientUpdateableQuadraticSVMBuilder getThis() {
			return this;
		}

		/**
		 * Sets the gamma.
		 * 
		 * @param gamma
		 *            the gamma
		 * @return the gradient updateable quadratic svm builder
		 */
		public GradientUpdateableQuadraticSVMBuilder setGamma(double gamma) {
			checkArgument(gamma > 0, "gamma must be positive. Given: %s", gamma);
			this.gamma = gamma;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(
				Configuration<GradientUpdateableQuadraticSVMBuilder> conf) {
			super.configure(conf);
			setGamma(conf.floatOptionFromShortName("G"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder#getConfiguration()
		 */
		@Override
		public Configuration<GradientUpdateableQuadraticSVMBuilder> getConfiguration() {
			Configuration<GradientUpdateableQuadraticSVMBuilder> conf = new Configuration<GradientUpdateableQuadraticSVMBuilder>(
					options);
			populateConfiguration(conf);

			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<GradientUpdateableQuadraticSVMBuilder> populateConfiguration(
				Configuration<GradientUpdateableQuadraticSVMBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addFloatValueOnShortName("G", gamma);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static GradientUpdateableQuadraticSVMOptions getOptions() {
			return new GradientUpdateableQuadraticSVMOptions();
		}

		/**
		 * The Class GradientUpdateableQuadraticSVMOptions.
		 */
		protected static class GradientUpdateableQuadraticSVMOptions
				extends
				GradientUpdateableOptions<GradienUpdateableQuadraticSVM, GradientUpdateableQuadraticSVMBuilder> {
			{
				addOption(new FloatOption(
						"G",
						"gamma",
						"gamma for quadratic SVM loss, lower values more closely approximate hinge loss",
						0.5, true, new GreaterThanValueBound(0),
						new LessThanOrEqualsValueBound(BIGVAL)));
			}

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.QSVM;
			}

		}
	}

	public static class GradientUpdateableModifiedHuberBuilder
			extends
			GradientUpdateableClassifierConfigurableBuilder<GradientUpdateableModifiedHuber, GradientUpdateableModifiedHuberBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 6818228602046314281L;

		/** The options. */
		public static OptionSet<GradientUpdateableModifiedHuberBuilder> options = new GradientUpdateableModifiedHuberOptions();

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param config
		 *            the config
		 */
		public GradientUpdateableModifiedHuberBuilder(
				Configuration<GradientUpdateableModifiedHuberBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public GradientUpdateableModifiedHuberBuilder(int dimension,
				boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new {@link GradientUpdateableModifiedHuberBuilder}
		 * dimension and bias must be set manually using setters
		 */
		public GradientUpdateableModifiedHuberBuilder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public GradientUpdateableModifiedHuber build() {
			GradientUpdateableModifiedHuber model = new GradientUpdateableModifiedHuber(
					initializeGradientBuilder().setGradientTruncationBuilder(
							truncationBuilder), getDimension(), bias);
			model.setPasses(passes)
					.setSmoothertype(regType)
					.setCrossvalidateSmootherTraining(
							crossValidateSmootherTraining).initialize();
			return model;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected GradientUpdateableModifiedHuberBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(
				Configuration<GradientUpdateableModifiedHuberBuilder> conf) {
			super.configure(conf);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder#getConfiguration()
		 */
		@Override
		public Configuration<GradientUpdateableModifiedHuberBuilder> getConfiguration() {
			Configuration<GradientUpdateableModifiedHuberBuilder> conf = new Configuration<GradientUpdateableModifiedHuberBuilder>(
					options);
			populateConfiguration(conf);

			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<GradientUpdateableModifiedHuberBuilder> populateConfiguration(
				Configuration<GradientUpdateableModifiedHuberBuilder> conf) {
			super.populateConfiguration(conf);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static GradientUpdateableModifiedHuberOptions getOptions() {
			return new GradientUpdateableModifiedHuberOptions();
		}

		/**
		 * The Class GradientUpdateableModifiedHuberOptions.
		 */
		protected static class GradientUpdateableModifiedHuberOptions
				extends
				GradientUpdateableOptions<GradientUpdateableModifiedHuber, GradientUpdateableModifiedHuberBuilder> {
			{

			}

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.HUBER;
			}

		}
	}

	public static class GradientUpdateableLogisticRegressionBuilder
			extends
			GradientUpdateableClassifierConfigurableBuilder<GradientUpdateableLogisticRegression, GradientUpdateableLogisticRegressionBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 6818228602046314281L;

		/** The options. */
		public static OptionSet<GradientUpdateableLogisticRegressionBuilder> options = new GradientUpdateableLogisticRegressionOptions();

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param config
		 *            the config
		 */
		public GradientUpdateableLogisticRegressionBuilder(
				Configuration<GradientUpdateableLogisticRegressionBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new gradient updateable quadratic svm builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public GradientUpdateableLogisticRegressionBuilder(int dimension,
				boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new
		 * {@link GradientUpdateableLogisticRegressionBuilder} dimension and
		 * bias must be set manually using setters
		 */
		public GradientUpdateableLogisticRegressionBuilder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public GradientUpdateableLogisticRegression build() {
			GradientUpdateableLogisticRegression model = new GradientUpdateableLogisticRegression(
					initializeGradientBuilder().setGradientTruncationBuilder(
							truncationBuilder), getDimension(), bias);
			model.setPasses(passes)
					.setSmoothertype(regType)
					.setCrossvalidateSmootherTraining(
							crossValidateSmootherTraining).initialize();
			return model;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected GradientUpdateableLogisticRegressionBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(
				Configuration<GradientUpdateableLogisticRegressionBuilder> conf) {
			super.configure(conf);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder#getConfiguration()
		 */
		@Override
		public Configuration<GradientUpdateableLogisticRegressionBuilder> getConfiguration() {
			Configuration<GradientUpdateableLogisticRegressionBuilder> conf = new Configuration<GradientUpdateableLogisticRegressionBuilder>(
					options);
			populateConfiguration(conf);

			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.optimizable.
		 * GradientUpdateableClassifierConfigurableBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<GradientUpdateableLogisticRegressionBuilder> populateConfiguration(
				Configuration<GradientUpdateableLogisticRegressionBuilder> conf) {
			super.populateConfiguration(conf);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static GradientUpdateableLogisticRegressionOptions getOptions() {
			return new GradientUpdateableLogisticRegressionOptions();
		}

		/**
		 * The Class GradientUpdateableModifiedHuberOptions.
		 */
		protected static class GradientUpdateableLogisticRegressionOptions
				extends
				GradientUpdateableOptions<GradientUpdateableLogisticRegression, GradientUpdateableLogisticRegressionBuilder> {
			{

			}

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.OPTLR;
			}

		}
	}
}
