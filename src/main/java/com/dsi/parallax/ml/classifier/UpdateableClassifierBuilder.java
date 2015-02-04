package com.dsi.parallax.ml.classifier;

import com.dsi.parallax.ml.classifier.smoother.Smoother;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.IntegerOption;

import java.util.Arrays;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * The Class UpdateableClassifierBuilder.
 *
 * @param <U> the generic type
 * @param <B> the generic type
 */
public abstract class UpdateableClassifierBuilder<U extends UpdateableClassifier<U>, B extends UpdateableClassifierBuilder<U, B>>
		extends ClassifierBuilder<U, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8451525497812078045L;
	
	/** The decay. */
	protected double weight = 1., decay = 1.1;
	
	/** The passes. */
	protected int passes = 1;

	/**
	 * Instantiates a new updateable classifier builder.
	 *
	 * @param dimension the dimension
	 * @param bias the bias
	 */
	protected UpdateableClassifierBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}
	
	/**
	 * Instantiates a new updateable classifier builder.
	 * the dimension and bias will need to be set manually
	 */
	protected UpdateableClassifierBuilder() {
		super();
	}

	/**
	 * Instantiates a new updateable classifier builder.
	 *
	 * @param config the config
	 */
	protected UpdateableClassifierBuilder(Configuration<B> config) {
		super(config);
		configure(config);
	}

	/**
	 * set the initial weight on updates, often denoted lambda.
	 *
	 * @param weight the weight
	 * @return builder object
	 */
	public B setWeight(double weight) {
		checkArgument(weight > 0, "weight must be positive. input: %s", weight);
		this.weight = weight;
		return thisBuilder;
	}

	/**
	 * set the periodic decay on weight, a multiplicative factor TODO: replace
	 * with setting an AnnealingSched.
	 *
	 * @param decay the decay
	 * @return the b
	 */
	public B setDecay(double decay) {
		checkArgument(decay > 1, "weight must be >1. input: %s", decay);
		this.decay = decay;
		return thisBuilder;
	}

	/**
	 * number of passes over the data when doing batchmode training.
	 *
	 * @param passes the passes
	 * @return the b
	 */
	public B setPasses(int passes) {
		checkArgument(passes > 0, "passes must be greater than 0. input: %s",
				passes);
		this.passes = passes;
		return thisBuilder;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.ClassifierBuilder#getConfiguration()
	 */
	@Override
	public Configuration<B> getConfiguration() {
		Configuration<B> conf = super.getConfiguration();
		return populateConfiguration(conf);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.ClassifierBuilder#populateConfiguration(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<B> populateConfiguration(Configuration<B> conf) {
		super.populateConfiguration(conf);
		conf.addIntegerValueOnShortName("p", passes);
		conf.addFloatValueOnShortName("w", weight);
		conf.addFloatValueOnShortName("c", decay);
		return conf;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.ClassifierBuilder#configure(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<B> conf) {
		super.configure(conf);
		checkArgument(
				Smoother.UPDATEABLE_SMOOTHERS.contains(regType),
				"%s is an invalid regularization type for updateable models. options are: %s",
				regType,
				Arrays.toString(Smoother.UPDATEABLE_SMOOTHERS.toArray()));
		setWeight(conf.floatOptionFromShortName("w"));
		setDecay(conf.floatOptionFromShortName("c"));
		setPasses(conf.integerOptionFromShortName("p"));
	}

	/**
	 * The Class UpdateableOptions.
	 *
	 * @param <U> the generic type
	 * @param <B> the generic type
	 */
	protected static abstract class UpdateableOptions<U extends AbstractUpdateableClassifier<U>, B extends UpdateableClassifierBuilder<U, B>>
			extends ClassifierOptions<U, B> {
		{
			addOption(new IntegerOption("p", "passes",
					"number of passes used for batch mode training using sgt",
					3, false, new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(2000)));
			addOption(new FloatOption("w", "weight",
					"weight on updates (usually used with SGD)", 1., false,
					new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("c", "decay",
					"decay for weight on updates (usually used with SGD)", 1.1,
					false, new GreaterThanValueBound(1),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new EnumOption<SmootherType>(
					"R",
					"regtype",
					false,
					"Regularization type. Options are:"
							+ Arrays.toString(Smoother.UPDATEABLE_SMOOTHERS
									.toArray()), SmootherType.class,
					SmootherType.NONE));
		}
	}
}
