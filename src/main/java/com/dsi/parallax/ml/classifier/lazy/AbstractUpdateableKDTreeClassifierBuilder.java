package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.classifier.UpdateableClassifierBuilder;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.IntegerOption;

import java.util.Arrays;

import static com.google.common.base.Preconditions.checkArgument;

public abstract class AbstractUpdateableKDTreeClassifierBuilder<U extends AbstractUpdateableKDTreeClassifier<U>, B extends UpdateableClassifierBuilder<U, B>>
		extends UpdateableClassifierBuilder<U, B> {
	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 7000465644943456538L;

	/** The k. */
	protected int k = 3;

	/** The kd type. */
	protected KDType kdType = KDType.EUCLIDIAN;

	/** The mixing. */
	protected KNNMixingType mixing = KNNMixingType.MEAN;

	/** The size limit. */
	protected int sizeLimit = Integer.MAX_VALUE;

	/**
	 * Instantiates a new sequential knn builder.
	 * 
	 * @param config
	 *            the config
	 */
	public AbstractUpdateableKDTreeClassifierBuilder(Configuration<B> config) {
		super(config);
		configure(config);
	}

	/**
	 * instantiates a new AbstractUpdateableKDTreeClassifierBuilder
	 * dimension and bias will need to be set manually
	 */
	public AbstractUpdateableKDTreeClassifierBuilder() {
		super();
	}
	
	/**
	 * Instantiates a new sequential knn builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public AbstractUpdateableKDTreeClassifierBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Sets the kd tree type.
	 * 
	 * @param type
	 *            the type
	 * @return the sequential knn builder
	 */
	public B setKDTreeType(KDType type) {
		this.kdType = type;
		return thisBuilder;
	}

	/**
	 * Sets the label mizing type.
	 * 
	 * @param mixingType
	 *            the mixing type
	 * @return the sequential knn builder
	 */
	public B setLabelMizingType(KNNMixingType mixingType) {
		this.mixing = mixingType;
		return thisBuilder;
	}

	/**
	 * Sets the k.
	 * 
	 * @param k
	 *            the k
	 * @return the sequential knn builder
	 */
	public B setK(int k) {
		checkArgument(k >= 0, "k must be > 0 input: %s", k);
		this.k = k;
		return thisBuilder;
	}

	/**
	 * Sets the size limit.
	 * 
	 * @param sizeLimit
	 *            the size limit
	 * @return the sequential knn builder
	 */
	public B setSizeLimit(int sizeLimit) {
		checkArgument(sizeLimit >= 100, "sizeLimit must be >= 100, given: %s",
				sizeLimit);
		this.sizeLimit = sizeLimit;
		return thisBuilder;
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
		setKDTreeType((KDType) conf.enumFromShortName("T"));
		setLabelMizingType((KNNMixingType) conf.enumFromShortName("M"));
		setK(conf.integerOptionFromShortName("k"));
		setSizeLimit(conf.integerOptionFromShortName("S"));

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
		conf.addEnumValueOnShortName("T", kdType);
		conf.addEnumValueOnShortName("M", mixing);
		conf.addIntegerValueOnShortName("k", k);
		conf.addIntegerValueOnShortName("S", sizeLimit);
		return conf;
	}

	/**
	 * The Class SequentialKNNOptions.
	 */
	protected abstract static class AbstractUpdateableKDTreeClassifierOptions<U extends AbstractUpdateableKDTreeClassifier<U>, B extends UpdateableClassifierBuilder<U, B>>
			extends UpdateableOptions<U, B> {
		{
			addOption(new EnumOption<KDType>("T", "kdtype", false,
					"distance metric to be used for kd tree: "
							+ Arrays.toString(KDType.values()), KDType.class,
					KDType.EUCLIDIAN));
			addOption(new EnumOption<KNNMixingType>("M", "mixing", false,
					"metric used for computing labels from neighbor's labels. options:"
							+ Arrays.toString(KNNMixingType.values()),
					KNNMixingType.class, KNNMixingType.MEAN));
			addOption(new IntegerOption("k", "k", "k to be used for knn", 10,
					true, new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(10000)));
			addOption(new IntegerOption("S", "size", "size limit for kdTree",
					10000, false, new GreaterThanOrEqualsValueBound(100),
					new LessThanOrEqualsValueBound(BIGVAL)));
		}

	}
}
