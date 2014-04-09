package com.dsi.parallax.ml.classifier.kernelmethods;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.UpdateableClassifierBuilder;
import com.dsi.parallax.ml.evaluation.LossGradientType;
import com.dsi.parallax.ml.mercerkernels.KernelConfigurableBuilder;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.ConfigurableOption;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.IntegerOption;
import com.dsi.parallax.ml.util.option.OptionSet;

/**
 * Builder for Updateable kernel-based classifeirs
 * 
 * @param <K>
 *            the concrete type of Kernel Classifier
 * @param <B>
 *            the concrete type of {@link KernelMethodBuilder}, used for method
 *            chaining
 */
public abstract class KernelMethodBuilder<K extends AbstractUpdateableKernelClassifier<K>, B extends KernelMethodBuilder<K, B>>
		extends UpdateableClassifierBuilder<K, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -299792388164532028L;

	/**
	 * The kernel builder, used for configuring the kernel function used for
	 * kenrel classifiers.
	 */
	protected KernelConfigurableBuilder kernelBuilder = new KernelConfigurableBuilder();

	/** This builder, used for method chaining. */
	public KernelSGDBuilder thisBuider;

	/**
	 * Instantiates a new kernel method builder.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected KernelMethodBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}
	
	/**
	 * Instantiates a new kernel method builder.
	 * the dimension and bias will need to be set manually
	 * using setter methods
	 */
	protected  KernelMethodBuilder() {
		super();
	}

	/**
	 * Instantiates a new kernel method builder.
	 * 
	 * @param config
	 *            the configuration describing the desired model settings
	 */
	protected KernelMethodBuilder(Configuration<B> config) {
		super(config);
	}

	/**
	 * Sets the kernel builder.
	 * 
	 * @param kernelBuilder
	 *            the kernel builder
	 * @return the b
	 */
	public B setKernelBuilder(KernelConfigurableBuilder kernelBuilder) {
		this.kernelBuilder = kernelBuilder;
		return thisBuilder;
	}

	/**
	 * Sets the kernel builder.
	 * 
	 * @param kernelConfig
	 *            the kernel config
	 * @return the b
	 */
	public B setKernelBuilder(
			Configuration<KernelConfigurableBuilder> kernelConfig) {
		return setKernelBuilder(new KernelConfigurableBuilder(kernelConfig));
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
		Configuration<KernelConfigurableBuilder> kernelBuilderConf = (Configuration<KernelConfigurableBuilder>) conf
				.configurationFromShortName("K");
		KernelConfigurableBuilder kernelConfigurable = new KernelConfigurableBuilder(
				kernelBuilderConf);
		setKernelBuilder(kernelConfigurable);
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
		conf.addConfigurableValueOnShortName("K",
				kernelBuilder.getConfiguration());
		return conf;
	}

	/**
	 * Options general to all Updateable Kernel Classifier Builders <br>
	 * settings:<br>
	 * 
	 * K/kerneloptions : [configuration] - options for kernel constructors
	 * 
	 * @param <K>
	 *            the concrete type of Kernel Classifier
	 * @param <B>
	 *            the concrete type of {@link KernelMethodBuilder}, used for
	 *            method chaining
	 */
	protected abstract static class UpdateableKernelOptions<K extends AbstractUpdateableKernelClassifier<K>, B extends KernelMethodBuilder<K, B>>
			extends UpdateableOptions<K, B> {
		{
			addOption(new ConfigurableOption<KernelConfigurableBuilder>("K",
					"kerneloptions", true,
					"options for kernel constructors. info: "
							+ KernelConfigurableBuilder.optionInfo(),
					new Configuration<KernelConfigurableBuilder>(
							KernelConfigurableBuilder.options)));
		}
	}

	/**
	 * Builder for {@link BudgetKernelPerceptron}.
	 */
	public static class BudgetKernelPerceptronBuilder
			extends
			KernelMethodBuilder<BudgetKernelPerceptron, BudgetKernelPerceptronBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = 2138661273082573823L;

		/** The options. */
		public static OptionSet<BudgetKernelPerceptronBuilder> options = new BudgetKernelPerceptronOptions();

		/** The budget for support vectors. */
		private int N = 100;

		/** The margin describing tolerance to errors. */
		private double margin = 0.1;

		/**
		 * Instantiates a new budget kernel perceptron builder.
		 * 
		 * @param config
		 *            the configuration describing the desired model settings
		 */
		public BudgetKernelPerceptronBuilder(
				Configuration<BudgetKernelPerceptronBuilder> config) {
			super(config);
			configure(config);
		}
		
		/**
		 *  Instantiates a new budget kernel perceptron builder.
		 *  dimension and bias will need to be set manually,
		 *  using setters
		 */
		public BudgetKernelPerceptronBuilder() {
			super();
		}

		/**
		 * Instantiates a new budget kernel perceptron builder.
		 * 
		 * @param dimension
		 *            the number of features in the instantiated classifier
		 * @param bias
		 *            should the model have an additional (+1) intercept term?
		 */
		public BudgetKernelPerceptronBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public BudgetKernelPerceptron build() {
			BudgetKernelPerceptron model = new BudgetKernelPerceptron(
					getDimension(), bias);
			model.setKernel(kernelBuilder.buildKernel())
					.setMargin(margin)
					.setPoolSize(N)
					.setSmoothertype(regType)
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
		protected BudgetKernelPerceptronBuilder getThis() {
			return this;
		}

		/**
		 * margin for correct classification- errors within the boundary don't
		 * produce an update. larger values lead to less frequent updates, but
		 * models with less influence from noise
		 * 
		 * @param margin
		 *            The margin describing tolerance to errors, [0,1]
		 * @return the budget kernel perceptron builder (for method chaining)
		 */
		public BudgetKernelPerceptronBuilder setMargin(double margin) {
			checkArgument(margin >= 0 && margin <= 1,
					"margin must be between 0 and 1. input: %s", margin);
			this.margin = margin;
			return thisBuilder;
		}

		/**
		 * size of pool of support vectors. large values lead to more detailed
		 * models but require more system resources
		 * 
		 * @param n
		 *            The budget for support vectors.
		 * @return the budget kernel perceptron builder (for method chaining)
		 */
		public BudgetKernelPerceptronBuilder setPoolSize(int n) {
			checkArgument(N > 0,
					"number of instances must be positive, given %s", N);
			this.N = n;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#configure
		 * (com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<BudgetKernelPerceptronBuilder> conf) {
			super.configure(conf);
			setPoolSize(conf.integerOptionFromShortName("n"));
			setMargin(conf.floatOptionFromShortName("m"));

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * getConfiguration()
		 */
		@Override
		public Configuration<BudgetKernelPerceptronBuilder> getConfiguration() {
			Configuration<BudgetKernelPerceptronBuilder> conf = new Configuration<BudgetKernelPerceptronBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<BudgetKernelPerceptronBuilder> populateConfiguration(
				Configuration<BudgetKernelPerceptronBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addIntegerValueOnShortName("n", N);
			conf.addFloatValueOnShortName("m", margin);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static BudgetKernelPerceptronOptions getOptions() {
			return new BudgetKernelPerceptronOptions();
		}

		/**
		 * BudgetKernelPerceptronOptions.<br>
		 * options:<br>
		 * n/numinsts [int] (> 0), The budget for support vectors<br>
		 * m/margin [double] [0,1] The margin describing tolerance to errors,
		 * [0,1]<br>
		 */
		protected static class BudgetKernelPerceptronOptions
				extends
				UpdateableKernelOptions<BudgetKernelPerceptron, BudgetKernelPerceptronBuilder> {
			{
				addOption(new IntegerOption("n", "numinsts",
						"number of instances for kernel eval", 100, false,
						new GreaterThanOrEqualsValueBound(1),
						new LessThanOrEqualsValueBound(BIGVAL)));
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
				return Classifiers.BUDGETKERNELPERCEPTRON;
			}
		}
	}

	/**
	 * Builder for {@link Forgetron} classifiers
	 */
	public static class ForgetronBuilder extends
			KernelMethodBuilder<Forgetron, ForgetronBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -4688142046978072795L;

		/** The options. */
		public static OptionSet<ForgetronBuilder> options = new ForgetronOptions();

		/** The budget for support vectors */
		private int budget = 1000;

		/**
		 * Instantiates a new forgetron builder.
		 * 
		 * @param config
		 *            the configuration describing the desired model settings
		 */
		public ForgetronBuilder(Configuration<ForgetronBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new forgetron builder.
		 * 
		 * @param dimension
		 *            the number of features in the instantiated classifier
		 * @param bias
		 *            should the model have an additional (+1) intercept term?
		 */
		public ForgetronBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * Instantiates a new Forgetron Builder
		 * dimension and bias will need ot be set manually using setters
		 */
		public ForgetronBuilder() {
			super();
		}
		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public Forgetron build() {
			Forgetron model = new Forgetron(budget, bias);
			model.setBudget(budget)
					.setKernel(kernelBuilder.buildKernel())
					.setSmoothertype(regType)
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
		protected ForgetronBuilder getThis() {
			return this;
		}

		/**
		 * Sets the budget.
		 * 
		 * @param budget
		 *            The budget for support vectors
		 * @return the forgetron builder (for method chaining)
		 */
		public ForgetronBuilder setBudget(int budget) {
			checkArgument(budget >= 1, "budget must be positive, given: %s",
					budget);
			this.budget = budget;
			return thisBuilder;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#configure
		 * (com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<ForgetronBuilder> config) {
			super.configure(config);
			setBudget(config.integerOptionFromShortName("B"));

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * getConfiguration()
		 */
		@Override
		public Configuration<ForgetronBuilder> getConfiguration() {
			Configuration<ForgetronBuilder> conf = new Configuration<ForgetronBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<ForgetronBuilder> populateConfiguration(
				Configuration<ForgetronBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addIntegerValueOnShortName("B", budget);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static ForgetronOptions getOptions() {
			return new ForgetronOptions();
		}

		/**
		 * Options for {@link ForgetronBuilder}, values are:<br>
		 * B/budget : [integer] The budget for support vectors
		 */
		protected static class ForgetronOptions extends
				UpdateableKernelOptions<Forgetron, ForgetronBuilder> {
			{
				addOption(new IntegerOption(
						"B",
						"budget",
						"budget for support vectors, number of examples stored",
						1000, false, new GreaterThanOrEqualsValueBound(1),
						new LessThanOrEqualsValueBound(5000)));
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
				return Classifiers.FORGETRON;
			}
		}
	}

	/**
	 * Builder for {@link KernelSGD} classifiers
	 */
	public static class KernelSGDBuilder extends
			KernelMethodBuilder<KernelSGD, KernelSGDBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -6363272402029766031L;

		/** The options. */
		public static OptionSet<KernelSGDBuilder> options = new KernelSGDOptions();

		/** The threshold for hinge loss */
		private double margin = 1.;

		/** aggressiveness of updates. */
		private double eta = 1;
		/** weight on regularization */
		private double lambda = 0.01;

		/** The loss function used */
		private LossGradientType lossType = LossGradientType.HINGELOSS;

		/**
		 * Instantiates a new kernel sgd builder.
		 * 
		 * @param config
		 *            the configuration describing the desired model settings
		 */
		public KernelSGDBuilder(Configuration<KernelSGDBuilder> config) {
			super(config);
			configure(config);
			thisBuider = getThis();
		}

		/**
		 * Instantiates a new kernel sgd builder.
		 * 
		 * @param dimension
		 *            the number of features in the instantiated classifier
		 * @param bias
		 *            should the model have an additional (+1) intercept term?
		 */
		public KernelSGDBuilder(int dimension, boolean bias) {
			super(dimension, bias);
			thisBuider = getThis();
		}
		
		/**
		 * Instantiates a kernel sgd builder
		 * dimension and bias will need to be set manually.
		 */
		public KernelSGDBuilder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public KernelSGD build() {
			KernelSGD model = new KernelSGD(getDimension(), bias);
			model.setEta(eta)
					.setLambda(lambda)
					.setMargin(margin)
					.setLossGradientType(lossType)
					.setKernel(kernelBuilder.buildKernel())
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
		protected KernelSGDBuilder getThis() {
			return this;
		}

		/**
		 * Sets the loss gradient type, the type of loss function to minimize
		 * 
		 * @param gradType
		 *            the type of loss to minimize
		 * @return the kernel sgd builder (for method chaining)
		 */
		public KernelSGDBuilder setLossGradientType(LossGradientType gradType) {
			this.lossType = gradType;
			return thisBuider;
		}

		/**
		 * Sets the threshhold used for hinge loss (dont try to optimize losses
		 * less than this)
		 * 
		 * @param thresh
		 *            the threshold for hinge loss
		 * @return the kernel sgd builder (for method chaining)
		 */
		public KernelSGDBuilder setMargin(double thresh) {
			checkArgument(thresh > 0, "margin must be > 0, given: %s", thresh);
			this.margin = thresh;
			return thisBuider;
		}

		/**
		 * Sets the eta, aggressiveness of updates
		 * 
		 * @param eta
		 *            aggressiveness of updates
		 * @return the kernel sgd builder (for method chaining)
		 */
		public KernelSGDBuilder setEta(double eta) {
			checkArgument(eta > 0, "eta must be > 0, given: %s", eta);
			this.eta = eta;
			return thisBuider;
		}

		/**
		 * Sets the lambda, weight on regularization
		 * 
		 * @param lambda
		 *            the weight on regularization
		 * @return the kernel sgd builder (for method chaining)
		 */
		public KernelSGDBuilder setLambda(double lambda) {
			checkArgument(lambda > 0, "lambda must be > 0, given: %s", lambda);
			this.lambda = lambda;
			return thisBuider;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#configure
		 * (com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public void configure(Configuration<KernelSGDBuilder> conf) {
			super.configure(conf);
			setLossGradientType((LossGradientType) conf.enumFromShortName("L"));
			setMargin(conf.floatOptionFromShortName("m"));
			setEta(conf.floatOptionFromShortName("E"));
			setLambda(conf.floatOptionFromShortName("M"));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * getConfiguration()
		 */
		@Override
		public Configuration<KernelSGDBuilder> getConfiguration() {
			Configuration<KernelSGDBuilder> conf = new Configuration<KernelSGDBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.kernelmethods.KernelMethodBuilder#
		 * populateConfiguration(com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<KernelSGDBuilder> populateConfiguration(
				Configuration<KernelSGDBuilder> conf) {
			super.populateConfiguration(conf);
			conf.addEnumValueOnShortName("L", lossType);
			conf.addFloatValueOnShortName("m", margin);
			conf.addFloatValueOnShortName("E", eta);
			conf.addFloatValueOnShortName("M", lambda);
			return conf;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static KernelSGDOptions getOptions() {
			return new KernelSGDOptions();
		}

		/**
		 * Options for {@link KernelMethodBuilder}, settings are:<br>
		 * L/losstype: type of loss function to minimize<br>
		 * H/hingethresh: float, >0, threshold for hinge loss<br>
		 * E/eta: float > 0, aggressiveness of updates<br>
		 * M/lambda: float > 0, weight on regularization<br>
		 */
		protected static class KernelSGDOptions extends
				UpdateableKernelOptions<KernelSGD, KernelSGDBuilder> {
			{
				addOption(new EnumOption<LossGradientType>("L", "losstype",
						true, "the type of loss to minimize, options: "
								+ Arrays.toString(LossGradientType.values()),
						LossGradientType.class, LossGradientType.HINGELOSS));
				addOption(new FloatOption("m", "margin",
						"threshold for hinge loss", 1, false,
						new GreaterThanValueBound(0),
						new LessThanOrEqualsValueBound(10000)));
				addOption(new FloatOption("E", "eta",
						"aggressiveness of updates", 1, false,
						new GreaterThanValueBound(0),
						new LessThanOrEqualsValueBound(10000)));
				addOption(new FloatOption("M", "lambda",
						"weight on regularization", 0.01, true,
						new GreaterThanValueBound(0),
						new LessThanOrEqualsValueBound(1)));
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
				return Classifiers.KERNELSGD;
			}
		}

	}

}
