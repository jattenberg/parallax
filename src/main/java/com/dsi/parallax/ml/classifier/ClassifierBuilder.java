package com.dsi.parallax.ml.classifier;

import java.util.Arrays;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.model.ModelBuilder;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.IntegerOption;

/**
 * The Base class for Classifier Builders.
 * 
 * @param <C>
 *            Type Concrete type of classifier to be built.
 * @param <B>
 *            The concrete type of builder, used for method chaining.
 */
public abstract class ClassifierBuilder<C extends Classifier<C>, B extends ClassifierBuilder<C, B>>
		extends ModelBuilder<BinaryClassificationTarget, C, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 7311290373227304009L;

	/**
	 * The type of smoother used for smoothing raw classifier scores into
	 * probability estimates
	 */
	protected SmootherType regType = SmootherType.NONE;

	/**
	 * when training smoother for probability smoothing, use cross validation
	 */
	protected int crossValidateSmootherTraining = 1;

	/**
	 * Instantiates a new classifier builder.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected ClassifierBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Instantiates a new classifier builder.
	 * 
	 * the bais and dimension will need to be set manually
	 */
	public ClassifierBuilder() {
		super();
	}

	/**
	 * Instantiates a new classifier builder using a configuration describing
	 * the settings of the member variables used for building models.
	 * 
	 * @param config
	 *            the config describing the desired setting of the classifier
	 */
	protected ClassifierBuilder(Configuration<B> config) {
		super(config);
		configure(config);
	}

	/**
	 * Sets the type of probability smoother to be used.
	 * 
	 * @param smoothertype
	 *            the smoother type
	 * @return the classifier builder
	 */
	public B setRegulizerType(SmootherType regType) {
		this.regType = regType;
		return thisBuilder;
	}

	/**
	 * determines if cross validation should be used when training
	 * 
	 * @return true if crossvalidation is used, else false.
	 */
	public int getCrossvalidateSmootherTraining() {
		return crossValidateSmootherTraining;
	}

	/**
	 * determines if cross validation should be used when training
	 * Smoother
	 * 
	 * @param crossvalidateSmootherTraining
	 * @return the classification model
	 */
	public B setCrossvalidateSmootherTraining(
			int crossvalidateSmootherTraining) {
		this.crossValidateSmootherTraining = crossvalidateSmootherTraining;
		return thisBuilder;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.option.Configurable#configure(com.parallax.ml.util
	 * .option.Configuration)
	 */
	@Override
	public void configure(Configuration<B> conf) {
		setRegulizerType((SmootherType) conf.enumFromShortName("R"));
		setCrossvalidateSmootherTraining(conf
				.integerOptionFromShortName("X"));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#getConfiguration()
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
	 * com.parallax.ml.model.ModelBuilder#populateConfiguration(com.parallax
	 * .ml.util.option.Configuration)
	 */
	@Override
	public Configuration<B> populateConfiguration(Configuration<B> conf) {
		super.populateConfiguration(conf);
		conf.addEnumValueOnShortName("R", regType);
		conf.addIntegerValueOnShortName("X",
				crossValidateSmootherTraining);
		return conf;
	}

	/**
	 * Base class for options for classifier types used for configuring
	 * classifier builders.
	 * 
	 * Defines the following options: S / smoothertype [{@link SmootherType}],
	 * "the type of smoother used"</br> X / xvalidate
	 * "when training smoother for probability smoothing, use cross validation"
	 * 
	 * @param <C>
	 *            The concrete type of classifier being constructed
	 * @param <B>
	 *            The type of builder, used for method chaining.
	 */
	protected static abstract class ClassifierOptions<C extends AbstractClassifier<C>, B extends ClassifierBuilder<C, B>>
			extends ModelOptions<BinaryClassificationTarget, C, B> {
		{
			addOption(new EnumOption<SmootherType>("S", "smoothertype", false,
					"Smoother type. Options are:"
							+ Arrays.toString(SmootherType.values()),
					SmootherType.class, SmootherType.NONE));
			addOption(new IntegerOption(
					"X",
					"xvalidate",
					"when training smoothers for probability smoothing, use cross validation",
					1, false, new GreaterThanOrEqualsValueBound(1)));
		}

		/**
		 * Gets the classifier type.
		 * 
		 * @return the classifier type
		 */
		public abstract Classifiers getClassifierType();

	}

}
