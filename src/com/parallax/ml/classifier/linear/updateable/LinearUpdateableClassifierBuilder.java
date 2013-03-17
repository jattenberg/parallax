package com.parallax.ml.classifier.linear.updateable;

import java.util.Arrays;

import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.UpdateableClassifier;
import com.parallax.ml.classifier.UpdateableClassifierBuilder;
import com.parallax.ml.classifier.UpdateableType;
import com.parallax.ml.classifier.linear.LinearClassifier;
import com.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.parallax.ml.util.bounds.GreaterThanValueBound;
import com.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.parallax.ml.util.bounds.LessThanValueBound;
import com.parallax.ml.util.option.BooleanOption;
import com.parallax.ml.util.option.ConfigurableOption;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.EnumOption;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.IntegerOption;
import com.parallax.ml.util.option.OptionSet;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;

// TODO: Auto-generated Javadoc
/**
 * builders for linear updateable models provides easy configuration of model
 * parameters, and construction from models with the associated parameters set.
 * 
 * @param <U>
 *            type of model used
 * @param <B>
 *            builder type
 * 
 *            TODO: better handling of gradient truncation
 * @author jattenberg
 */
public abstract class LinearUpdateableClassifierBuilder<U extends LinearClassifier<U> & UpdateableClassifier<U>, B extends LinearUpdateableClassifierBuilder<U, B>>
		extends UpdateableClassifierBuilder<U, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6910146537188102731L;

	/** The thresh. */
	protected double thresh = 0;

	/** The regularization weight. */
	protected double gaussianWeight = 0, laplaceWeight = 0, cauchyWeight = 0,
			squaredWeight = 0, regularizationWeight = 1;

	/** The regularize intercept. */
	protected boolean regularizeIntercept = false;

	/** The truncation builder. */
	protected TruncationConfigurableBuilder truncationBuilder = new TruncationConfigurableBuilder();
	
	protected AnnealingSchedule annealingSchedule = new ConstantAnnealingSchedule(0.1);

	/**
	 * Instantiates a new linear updateable classifier builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	protected LinearUpdateableClassifierBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Instantiates a new linear updateable classifier builder.
	 * 
	 * @param config
	 *            the config
	 */
	protected LinearUpdateableClassifierBuilder(Configuration<B> config) {
		super(config);
		configure(config);
	}

	/**
	 * instantiates a new LinearUpdateableClassifierBuilder dimension and bias
	 * will need to be set manually using setters
	 */
	protected LinearUpdateableClassifierBuilder() {
		super();
	}

	/**
	 * Sets the annealing schedule.
	 * 
	 * @param annealingSchedule
	 *            the annealing schedule
	 * @return the b
	 */
	public B setAnnealingSchedule(AnnealingSchedule annealingSchedule) {
		this.annealingSchedule = annealingSchedule;
		return thisBuilder;
	}
	
	/**
	 * Sets the gaussian weight.
	 * 
	 * @param gaussianWeight
	 *            the gaussian weight
	 * @return the b
	 */
	public B setGaussianWeight(double gaussianWeight) {
		this.gaussianWeight = gaussianWeight;
		return thisBuilder;
	}

	/**
	 * Sets the laplace weight.
	 * 
	 * @param laplaceWeight
	 *            the laplace weight
	 * @return the b
	 */
	public B setLaplaceWeight(double laplaceWeight) {
		this.laplaceWeight = laplaceWeight;
		return thisBuilder;
	}

	/**
	 * Sets the cauchy weight.
	 * 
	 * @param cauchyWeight
	 *            the cauchy weight
	 * @return the b
	 */
	public B setCauchyWeight(double cauchyWeight) {
		this.cauchyWeight = cauchyWeight;
		return thisBuilder;
	}

	/**
	 * Sets the squared weight.
	 * 
	 * @param squaredWeight
	 *            the squared weight
	 * @return the b
	 */
	public B setSquaredWeight(double squaredWeight) {
		this.squaredWeight = squaredWeight;
		return thisBuilder;
	}

	/**
	 * Sets the regularize intercept.
	 * 
	 * @param regularizeIntercept
	 *            the regularize intercept
	 * @return the b
	 */
	public B setRegularizeIntercept(boolean regularizeIntercept) {
		this.regularizeIntercept = regularizeIntercept;
		return thisBuilder;
	}

	/**
	 * Sets the regularization weight.
	 * 
	 * @param regularizationWeight
	 *            the regularization weight
	 * @return the b
	 */
	protected B setRegularizationWeight(double regularizationWeight) {
		this.regularizationWeight = regularizationWeight;
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
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#configure(com.
	 * parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<B> conf) {
		super.configure(conf);

		// set up gradient truncation
		@SuppressWarnings("unchecked")
		Configuration<TruncationConfigurableBuilder> truncConfig = (Configuration<TruncationConfigurableBuilder>) conf
				.configurationFromShortName("T");
		setTruncationBuilder(truncConfig);

		// configure regularization
		setRegularizeIntercept(conf.booleanOptionFromShortName("i"));
		setRegularizationWeight(conf.floatOptionFromShortName("r"));

		setGaussianWeight(conf.floatOptionFromShortName("GR"));
		setLaplaceWeight(conf.floatOptionFromShortName("LR"));
		setCauchyWeight(conf.floatOptionFromShortName("CR"));
		setSquaredWeight(conf.floatOptionFromShortName("SR"));

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

		conf.addFloatValueOnShortName("r", regularizationWeight);

		conf.addFloatValueOnShortName("GR", gaussianWeight);
		conf.addFloatValueOnShortName("LR", laplaceWeight);
		conf.addFloatValueOnShortName("CR", cauchyWeight);
		conf.addFloatValueOnShortName("SR", squaredWeight);

		return conf;
	}

	/**
	 * The Class LinearOptions.
	 * 
	 * @param <C>
	 *            the generic type
	 * @param <B>
	 *            the generic type
	 */
	protected static abstract class LinearOptions<C extends AbstractLinearUpdateableClassifier<C>, B extends LinearUpdateableClassifierBuilder<C, B>>
			extends UpdateableOptions<C, B> {
		{
			addOption(new BooleanOption("i", "regularizeintercept",
					"apply regularization to the intercept term", false, true));
			addOption(new ConfigurableOption<TruncationConfigurableBuilder>(
					"T", "truncationConfig", false,
					"the configuration for builders of gradient truncations. Options: "
							+ TruncationConfigurableBuilder.optionInfo(),
					new Configuration<TruncationConfigurableBuilder>(
							TruncationConfigurableBuilder.options)));

			// regularization coefficients
			addOption(new FloatOption("GR", "gaussianWeight",
					"weight on gaussian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
			addOption(new FloatOption("LR", "laplaceWeight",
					"weight on laplacian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
			addOption(new FloatOption("CR", "cauchyWeight",
					"weight on cauchian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
			addOption(new FloatOption("SR", "squaredWeight",
					"weight on squared loss regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
			addOption(new FloatOption("r", "regweight",
					"weight on regularization", 1, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
		}
	}

	/**
	 * The Class WinnowClassifierBuilder.
	 */
	public static class WinnowClassifierBuilder
			extends
			LinearUpdateableClassifierBuilder<WinnowClassifier, WinnowClassifierBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 6477638330922770024L;

		/** The margin. */
		protected double margin = 0.5;

		/** The options. */
		public static OptionSet<WinnowClassifierBuilder> options = new WinnowOptions();

		/**
		 * Instantiates a new winnow classifier builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public WinnowClassifierBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new winnow classifier builder dimension and bias must
		 * be set manually using setters
		 */
		public WinnowClassifierBuilder() {
			super();
		}

		/**
		 * Instantiates a new winnow classifier builder.
		 * 
		 * @param config
		 *            the config
		 */
		public WinnowClassifierBuilder(
				Configuration<WinnowClassifierBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Sets the margin.
		 * 
		 * @param margin
		 *            the margin
		 * @return the winnow classifier builder
		 */
		public WinnowClassifierBuilder setMargin(double margin) {
			this.margin = margin;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public WinnowClassifier build() {
			WinnowClassifier model = new WinnowClassifier(getDimension(), bias);
			model.setMargin(margin)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setSmoothertype(regType)
					.setRegularizeIntercept(regularizeIntercept)
					.setPasses(passes)
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
		protected WinnowClassifierBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<WinnowClassifierBuilder> conf) {
			super.configure(conf);
			setMargin(conf.floatOptionFromShortName("m"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<WinnowClassifierBuilder> getConfiguration() {
			Configuration<WinnowClassifierBuilder> conf = new Configuration<WinnowClassifierBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<WinnowClassifierBuilder> populateConfiguration(
				Configuration<WinnowClassifierBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addFloatValueOnShortName("m", margin);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static WinnowOptions getOptions() {
			return new WinnowOptions();
		}

		/**
		 * The Class WinnowOptions.
		 */
		protected static class WinnowOptions extends
				LinearOptions<WinnowClassifier, WinnowClassifierBuilder> {
			{
				addOption(new FloatOption("m", "margin",
						"margin for determining correctness", 0., true,
						new GreaterThanOrEqualsValueBound(0),
						new LessThanValueBound(BIGVAL)));
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
				return Classifiers.WINNOW;
			}
		}
	}

	/**
	 * The Class PerceptronWithMarginBuilder.
	 */
	public static class PerceptronWithMarginBuilder
			extends
			LinearUpdateableClassifierBuilder<PerceptronWithMargin, PerceptronWithMarginBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 5291978301621817547L;

		/** The margin. */
		protected double margin = 0.5;

		/** The options. */
		public static OptionSet<PerceptronWithMarginBuilder> options = new PercptronWithMarginsOptions();

		/**
		 * Instantiates a new perceptron with margin builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public PerceptronWithMarginBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * Instantiates a new PerceptronWithMarginBuilder dimension and bias
		 * must be set manually using setters
		 */
		public PerceptronWithMarginBuilder() {
			super();
		}

		/**
		 * Instantiates a new perceptron with margin builder.
		 * 
		 * @param config
		 *            the config
		 */
		public PerceptronWithMarginBuilder(
				Configuration<PerceptronWithMarginBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Sets the margin.
		 * 
		 * @param margin
		 *            the margin
		 * @return the perceptron with margin builder
		 */
		public PerceptronWithMarginBuilder setMargin(double margin) {
			this.margin = margin;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public PerceptronWithMargin build() {
			PerceptronWithMargin model = new PerceptronWithMargin(
					getDimension(), bias);
			model.setMargin(margin)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setSmoothertype(regType)
					.setRegularizeIntercept(regularizeIntercept)
					.setPasses(passes)
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
		protected PerceptronWithMarginBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<PerceptronWithMarginBuilder> conf) {
			super.configure(conf);
			setMargin(conf.floatOptionFromShortName("m"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<PerceptronWithMarginBuilder> getConfiguration() {
			Configuration<PerceptronWithMarginBuilder> conf = new Configuration<PerceptronWithMarginBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<PerceptronWithMarginBuilder> populateConfiguration(
				Configuration<PerceptronWithMarginBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addFloatValueOnShortName("m", margin);
			return conf;
		}

		/**
		 * Gets the margin.
		 * 
		 * @return the margin
		 */
		public double getMargin() {
			return margin;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static PercptronWithMarginsOptions getOptions() {
			return new PercptronWithMarginsOptions();
		}

		/**
		 * The Class PercptronWithMarginsOptions.
		 */
		protected static class PercptronWithMarginsOptions
				extends
				LinearOptions<PerceptronWithMargin, PerceptronWithMarginBuilder> {
			{
				addOption(new FloatOption("m", "margin",
						"margin for determining correctness", 0., true,
						new GreaterThanOrEqualsValueBound(0),
						new LessThanValueBound(1)));
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
				return Classifiers.MARGINPERCEPTRON;
			}
		}

	}

	/**
	 * The Class PegasosBuilder.
	 */
	public static class PegasosBuilder extends
			LinearUpdateableClassifierBuilder<Pegasos, PegasosBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 721678444332659300L;

		/** The window size. */
		int windowSize = 1;
		
		private double regularizationWeight = 0.01;

		/** The options. */
		public static OptionSet<PegasosBuilder> options = new PegasosOptions();

		/**
		 * Instantiates a new pegasos builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public PegasosBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * Instantiates a new pegasos builder.
		 * 
		 * @param config
		 *            the config
		 */
		public PegasosBuilder(Configuration<PegasosBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new Pegasos Builder dimension and bias must be set
		 * manually using setters
		 */
		public PegasosBuilder() {
			super();
		}

		/**
		 * Sets the window size.
		 * 
		 * @param windowSize
		 *            the window size
		 * @return the pegasos builder
		 */
		public PegasosBuilder setWindowSize(int windowSize) {
			this.windowSize = windowSize;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public Pegasos build() {
			Pegasos model = new Pegasos(getDimension(), bias);
			model.setWindowSize(windowSize)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setPasses(passes)
					.setRegularizeIntercept(regularizeIntercept)
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
		protected PegasosBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<PegasosBuilder> conf) {
			super.configure(conf);
			setWindowSize(conf.integerOptionFromShortName("k"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<PegasosBuilder> getConfiguration() {
			Configuration<PegasosBuilder> conf = new Configuration<PegasosBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<PegasosBuilder> populateConfiguration(
				Configuration<PegasosBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addIntegerValueOnShortName("k", windowSize);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static PegasosOptions getOptions() {
			return new PegasosOptions();
		}

		/**
		 * The Class PegasosOptions.
		 */
		protected static class PegasosOptions extends
				LinearOptions<Pegasos, PegasosBuilder> {
			{
				addOption(new IntegerOption("k", "windowsize",
						"window size for pegasos", 1, false,
						new GreaterThanOrEqualsValueBound(0),
						new LessThanOrEqualsValueBound(1000)));
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
				return Classifiers.PEGASOS;
			}
		}

	}

	/**
	 * The Class PassiveAggressiveBuilder.
	 */
	public static class PassiveAggressiveBuilder
			extends
			LinearUpdateableClassifierBuilder<PassiveAggressive, PassiveAggressiveBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 8829290373279093492L;

		/** The options. */
		public static OptionSet<PassiveAggressiveBuilder> options = new PAOptions();

		/** The aggressiveness. */
		private double aggressiveness = 0.3;

		/**
		 * Instantiates a new passive aggressive builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public PassiveAggressiveBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new PassiveAggressiveBuilder dimension and bias must
		 * be set manually using setters
		 */
		public PassiveAggressiveBuilder() {
			super();
		}

		/**
		 * Instantiates a new passive aggressive builder.
		 * 
		 * @param config
		 *            the config
		 */
		public PassiveAggressiveBuilder(
				Configuration<PassiveAggressiveBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Sets the aggressiveness.
		 * 
		 * @param aggressiveness
		 *            the aggressiveness
		 * @return the passive aggressive builder
		 */
		public PassiveAggressiveBuilder setAggressiveness(double aggressiveness) {
			this.aggressiveness = aggressiveness;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public PassiveAggressive build() {
			PassiveAggressive model = new PassiveAggressive(getDimension(),
					bias);
			model.setAggressiveness(aggressiveness)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setPasses(passes)
					.setRegularizeIntercept(regularizeIntercept)
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
		protected PassiveAggressiveBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<PassiveAggressiveBuilder> getConfiguration() {
			Configuration<PassiveAggressiveBuilder> conf = new Configuration<PassiveAggressiveBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<PassiveAggressiveBuilder> populateConfiguration(
				Configuration<PassiveAggressiveBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addFloatValueOnShortName("a", aggressiveness);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<PassiveAggressiveBuilder> conf) {
			super.configure(conf);
			setAggressiveness(conf.floatOptionFromShortName("a"));
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static PAOptions getOptions() {
			return new PAOptions();
		}

		/**
		 * The Class PAOptions.
		 */
		protected static class PAOptions extends
				LinearOptions<PassiveAggressive, PassiveAggressiveBuilder> {
			{
				addOption(new FloatOption("a", "aggressiveness",
						"aggressiveness for passive aggressive training", 0.3,
						false, new GreaterThanValueBound(0),
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
				return Classifiers.PASSIVEAGGRESSIVE;
			}
		}
	}

	/**
	 * The Class LogisticRegressionBuilder.
	 */
	public static class LogisticRegressionBuilder
			extends
			LinearUpdateableClassifierBuilder<LogisticRegression, LogisticRegressionBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -210935995172312472L;

		/** The shift. */
		private boolean shift = false;

		/** The gamma. */
		private double gamma = 1;

		/** The update. */
		private UpdateableType update = UpdateableType.COORDINATEWISE;

		/** The options. */
		public static OptionSet<LogisticRegressionBuilder> options = new LogisticRegressionOptions();

		/**
		 * Instantiates a new logistic regression builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public LogisticRegressionBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new logistic regression builder dimension and bias
		 * must be set manually using setters
		 */
		public LogisticRegressionBuilder() {
			super();
		}

		/**
		 * Instantiates a new logistic regression builder.
		 * 
		 * @param config
		 *            the config
		 */
		public LogisticRegressionBuilder(
				Configuration<LogisticRegressionBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Sets the updateable type.
		 * 
		 * @param type
		 *            the type
		 * @return the logistic regression builder
		 */
		public LogisticRegressionBuilder setUpdateableType(UpdateableType type) {
			this.update = type;
			return thisBuilder;
		}

		/**
		 * Sets the shift.
		 * 
		 * @param useShift
		 *            the use shift
		 * @return the logistic regression builder
		 */
		public LogisticRegressionBuilder setShift(boolean useShift) {
			this.shift = useShift;
			return thisBuilder;
		}

		/**
		 * Sets the gamma.
		 * 
		 * @param gamma
		 *            the gamma
		 * @return the logistic regression builder
		 */
		public LogisticRegressionBuilder setGamma(double gamma) {
			this.gamma = gamma;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public LogisticRegression build() {
			LogisticRegression model = new LogisticRegression(getDimension(),
					bias);
			model.setShift(shift)
					.setUpdateableType(update)
					.setGamma(gamma)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setSmoothertype(regType)
					.setPasses(passes)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setRegularizeIntercept(regularizeIntercept)
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
		protected LogisticRegressionBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<LogisticRegressionBuilder> conf) {
			super.configure(conf);
			setUpdateableType((UpdateableType) conf.enumFromShortName("u"));
			setShift(conf.booleanOptionFromShortName("f"));
			setGamma(conf.floatOptionFromShortName("g"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<LogisticRegressionBuilder> getConfiguration() {
			Configuration<LogisticRegressionBuilder> conf = new Configuration<LogisticRegressionBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<LogisticRegressionBuilder> populateConfiguration(
				Configuration<LogisticRegressionBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addEnumValueOnShortName("u", update);
			conf.addBooleanValueOnShortName("f", shift);
			conf.addFloatValueOnShortName("g", gamma);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static LogisticRegressionOptions getOptions() {
			return new LogisticRegressionOptions();
		}

		/**
		 * The Class LogisticRegressionOptions.
		 */
		protected static class LogisticRegressionOptions extends
				LinearOptions<LogisticRegression, LogisticRegressionBuilder> {
			{
				addOption(new BooleanOption("f", "shift",
						"shift the logistic loss by 1 (as in hinge loss)",
						false, true));

				addOption(new FloatOption("g", "gamma",
						"control on the sharpness of the loss function", 1,
						false, new GreaterThanValueBound(0),
						new LessThanOrEqualsValueBound(BIGVAL)));
				addOption(new EnumOption<UpdateableType>("u", "updatetype",
						false, "technique for setting rate for SGD, options: "
								+ Arrays.toString(UpdateableType.values()),
						UpdateableType.class, UpdateableType.COORDINATEWISE));
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
				return Classifiers.LOGISTICREGRESSION;
			}
		}
	}

	/**
	 * The Class AROWClassifierBuilder.
	 */
	public static class AROWClassifierBuilder
			extends
			LinearUpdateableClassifierBuilder<AROWClassifier, AROWClassifierBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -7294376621223754645L;

		/** The r. */
		private double R;

		/** The options. */
		public static OptionSet<AROWClassifierBuilder> options = new AROWOptions();

		/**
		 * Instantiates a new aROW classifier builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public AROWClassifierBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new AROWClassifierBuilder dimension and bias must be
		 * set manually using setters
		 */
		public AROWClassifierBuilder() {
			super();
		}

		/**
		 * Instantiates a new aROW classifier builder.
		 * 
		 * @param config
		 *            the config
		 */
		public AROWClassifierBuilder(Configuration<AROWClassifierBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Sets the r.
		 * 
		 * @param R
		 *            the r
		 * @return the aROW classifier builder
		 */
		public AROWClassifierBuilder setR(double R) {
			this.R = R;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public AROWClassifier build() {
			AROWClassifier model = new AROWClassifier(getDimension(), bias);
			model.setR(R)
					.setCauchyRegularizationWeight(cauchyWeight)
					.setGaussianRegularizationWeight(gaussianWeight)
					.setLaplaceRegularizationWeight(laplaceWeight)
					.setSquaredRegularizationWeight(squaredWeight)
					.setRegularizationWeight(regularizationWeight)
					.setTruncationBuilder(truncationBuilder)
					.setPasses(passes)
					.setRegularizeIntercept(regularizeIntercept)
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
		protected AROWClassifierBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #configure(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<AROWClassifierBuilder> conf) {
			super.configure(conf);
			setR(conf.floatOptionFromShortName("r"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<AROWClassifierBuilder> getConfiguration() {
			Configuration<AROWClassifierBuilder> conf = new Configuration<AROWClassifierBuilder>(
					options);
			return populateConfiguration(conf);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.linearupdateable.
		 * LinearUpdateableClassifierBuilder
		 * #populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<AROWClassifierBuilder> populateConfiguration(
				Configuration<AROWClassifierBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addFloatValueOnShortName("r", R);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static AROWOptions getOptions() {
			return new AROWOptions();
		}

		/**
		 * The Class AROWOptions.
		 */
		protected static class AROWOptions extends
				LinearOptions<AROWClassifier, AROWClassifierBuilder> {
			{
				addOption(new FloatOption(
						"r",
						"rvalue",
						"r value used for AROW algorithm, smaller values yield more aggressive updates",
						0., false, new GreaterThanOrEqualsValueBound(0),
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
				return Classifiers.AROW;
			}
		}

	}
}
