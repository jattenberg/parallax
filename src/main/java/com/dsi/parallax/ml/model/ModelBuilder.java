package com.dsi.parallax.ml.model;

import com.dsi.parallax.ml.target.Target;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.*;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * Base class for any model builders. implements setters for dimension and bias
 * terms.
 * 
 * The only required parameters are the model dimension and the presence of a
 * bias term All other parameters defined in subclasses should have default
 * values.
 * 
 * @param <T>
 *            The type of target being predicted and trained on.
 * @param <M>
 *            The concrete type of model being constructed
 * @param <B>
 *            The type of builder, used for method chaining.
 */
public abstract class ModelBuilder<T extends Target, M extends Model<T, M>, B extends ModelBuilder<T, M, B>>
		extends AbstractConfigurable<B> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3557204181735472998L;

	/** The dimension; number of features in the model. */
	private int dimension = -1;

	/**
	 * does the model have a bias term? if set to false, the model's hyperplane
	 * will pass through the origin.
	 */

	protected boolean bias = false;

	/** the builder itself; used for method chaining only. */
	protected final B thisBuilder;

	/**
	 * Instantiates a new model builder.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	protected ModelBuilder(int dimension, boolean bias) {
		this();
		checkArgument(dimension >= 0,
				"dimension must be non-negative. given: %s", dimension);
		this.dimension = dimension;
		this.bias = bias;
		
	}

	protected ModelBuilder() {
		thisBuilder = getThis();
	}
	
	/**
	 * Gets the dimension of the model.
	 * 
	 * @return the dimension
	 */
	public int getDimension() {
		checkArgument(dimension >= 0, "dimension not set");
		return dimension;
	}

	public B setDimension(int dimension) {
		checkArgument(dimension >= 0,
				"dimension must be non-negative. given: %s", dimension);
		this.dimension = dimension;
		return thisBuilder;
	}

	/**
	 * Gets the bias, should the model have an additional (+1) intercept term?.
	 * 
	 * @return the bias, true if the model should the model have an additional
	 *         (+1) intercept term
	 */
	public boolean getBias() {
		return bias;
	}
	
	/**
	 * Gets the bias, should the model have an additional (+1) intercept term?.
	 * 
	 * @return the bias, true if the model should the model have an additional
	 *         (+1) intercept term
	 */
	public B setBias(boolean bias) {
		this.bias = bias;
		return thisBuilder;
	}

	/**
	 * Instantiates a new model builder.
	 * 
	 * @param conf
	 *            the conf
	 */
	public ModelBuilder(Configuration<B> conf) {
		checkArgument(conf.containsShortKey("d"),
				"missing required short name: %s", "d");
		checkArgument(conf.containsShortKey("b"),
				"missing required short name: %s", "b");
		bias = conf.booleanOptionFromShortName("b");
		dimension = conf.integerOptionFromShortName("d");
		thisBuilder = getThis();
	}

	/**
	 * Builds the model using the set parameters.
	 * 
	 * @return A fresh instance the concrete model with all parameters set.
	 */
	public abstract M build();

	/**
	 * Gets the model builder. All concrete implementations should implement and
	 * (return this)
	 * 
	 * @return the model builder itself. Used for method chaining.
	 */
	protected abstract B getThis();

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.option.Configurable#getConfiguration()
	 */
	@Override
	public Configuration<B> getConfiguration() {
		Configuration<B> conf = new Configuration<B>(
				new ModelOptions<T, M, B>());
		return populateConfiguration(conf);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.option.Configurable#populateConfiguration(com.parallax
	 * .ml .util.option.Configuration)
	 */
	@Override
	public Configuration<B> populateConfiguration(Configuration<B> conf) {
		conf.addBooleanValueOnShortName("b", bias);
		conf.addIntegerValueOnShortName("d", dimension);
		return conf;
	}

	/**
	 * Base class for options for model types. Used for configuring model
	 * builders defines the following options: d / dimension [integer]
	 * "input dimension of the model" b / bias [boolean]
	 * "consider an extra constant term to deal with non-zero intercepts"
	 * 
	 * @param <T>
	 *            The type of target being predicted and trained on.
	 * @param <M>
	 *            The concrete type of model being constructed
	 * @param <B>
	 *            The type of builder, used for method chaining.
	 */
	protected static class ModelOptions<T extends Target, M extends Model<T, M>, B extends ModelBuilder<T, M, B>>
			extends OptionSet<B> {
		{
			addOption(new IntegerOption("d", "dimension",
					"input dimension of the model", (int) Math.pow(2, 16),
					false, new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new BooleanOption(
					"b",
					"bias",
					"consider an extra constant term to deal with non-zero intercepts",
					false, false));
		}
	}
}
