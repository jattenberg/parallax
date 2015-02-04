package com.dsi.parallax.ml.classifier.trees;

import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.IntegerOption;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * The Class TreeClassifierBuilder.
 *
 * @param <C> the generic type
 * @param <B> the generic type
 */
public abstract class TreeClassifierBuilder<C extends AbstractTreeClassifier<C>, B extends TreeClassifierBuilder<C, B>>
		extends ClassifierBuilder<C, B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 6902320437310904483L;

	/** The prepruning attempts. */
	protected int maxDepth = Integer.MAX_VALUE, minExamples = 0,
			prepruningAttempts = 20;
	
	/** The projection ratio. */
	protected double minEntropy = 0, projectionRatio = 1.;

	/**
	 * Instantiates a new tree classifier builder.
	 *
	 * @param config the config
	 */
	public TreeClassifierBuilder(Configuration<B> config) {
		super(config);
		configure(config);
	}

	/**
	 * Instantiates a new tree classifier builder.
	 *
	 * @param dimension the dimension
	 * @param bias the bias
	 */
	public TreeClassifierBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * instantiates a new tree classifier
	 * dimension and bias must be set manually using setters
	 */
	public TreeClassifierBuilder() {
		super();
	}
	
	/**
	 * Sets the max depth.
	 *
	 * @param maxDepth the max depth
	 * @return the b
	 */
	public B setMaxDepth(int maxDepth) {
		checkArgument(maxDepth >= 0, "maxDepth must be positive, given %s",
				maxDepth);
		this.maxDepth = maxDepth;
		return thisBuilder;
	}

	/**
	 * Sets the min examples.
	 *
	 * @param minExamples the min examples
	 * @return the b
	 */
	public B setMinExamples(int minExamples) {
		checkArgument(minExamples >= 0,
				"minExamples must be positive, given %s", minExamples);
		this.minExamples = minExamples;
		return thisBuilder;
	}

	/**
	 * Sets the min entropy.
	 *
	 * @param minEntropy the min entropy
	 * @return the b
	 */
	public B setMinEntropy(double minEntropy) {
		checkArgument(minEntropy >= 0,
				"entropy must be non-negative. given: %s", minEntropy);
		this.minEntropy = minEntropy;
		return thisBuilder;
	}

	/**
	 * Sets the prepruning attempts.
	 *
	 * @param prepruningAttempts the prepruning attempts
	 * @return the b
	 */
	public B setPrepruningAttempts(int prepruningAttempts) {
		checkArgument(prepruningAttempts >= 0,
				"prepruningAttempts must be greater than 0, given %s",
				prepruningAttempts);
		this.prepruningAttempts = prepruningAttempts;
		return thisBuilder;
	}

	/**
	 * Sets the projection ratio.
	 *
	 * @param projectionRatio the projection ratio
	 * @return the b
	 */
	public B setProjectionRatio(double projectionRatio) {
		checkArgument(projectionRatio > 0,
				"projectionRatio must be greater than 0, given %s",
				projectionRatio);
		this.projectionRatio = projectionRatio;
		return thisBuilder;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.ClassifierBuilder#configure(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<B> configuration) {
		super.configure(configuration);
		setMaxDepth(configuration.integerOptionFromShortName("M"));
		setMinExamples(configuration.integerOptionFromShortName("m"));
		setMinEntropy(configuration.floatOptionFromShortName("e"));
		setPrepruningAttempts(configuration.integerOptionFromShortName("p"));
		setProjectionRatio(configuration.floatOptionFromShortName("r"));

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

		conf.addIntegerValueOnShortName("M", maxDepth);
		conf.addIntegerValueOnShortName("m", minExamples);
		conf.addFloatValueOnShortName("e", minEntropy);
		conf.addIntegerValueOnShortName("p", prepruningAttempts);
		conf.addFloatValueOnShortName("r", projectionRatio);
		return conf;
	}

	/**
	 * The Class TreeClassifierOptions.
	 *
	 * @param <C> the generic type
	 * @param <B> the generic type
	 */
	protected abstract static class TreeClassifierOptions<C extends AbstractTreeClassifier<C>, B extends TreeClassifierBuilder<C, B>>
			extends ClassifierOptions<C, B> {
		{
			addOption(new IntegerOption("M", "maxdepth",
					"maximum depth allowable in a tree", 10, true,
					new GreaterThanOrEqualsValueBound(0)));
			addOption(new IntegerOption(
					"m",
					"minexamples",
					"minimum number of examples that must be present in a node",
					50, true, new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("e", "minEntropy",
					"minimum allowable label entropy in a node", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new IntegerOption("p", "prepruning",
					"number of prepurning attempts to use at each split", 30,
					false, new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption(
					"r",
					"ratio",
					"dimension reduction ratio when doing internal projections",
					1., false, new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(5)));
		}
	}

}
