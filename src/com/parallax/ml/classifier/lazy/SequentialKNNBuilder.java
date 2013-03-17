package com.parallax.ml.classifier.lazy;

import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.OptionSet;

// TODO: Auto-generated Javadoc
/**
 * The Class SequentialKNNBuilder.
 */
public class SequentialKNNBuilder
		extends
		AbstractUpdateableKDTreeClassifierBuilder<SequentialKNN, SequentialKNNBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 7000465644943456538L;

	/** The options. */
	public static OptionSet<SequentialKNNBuilder> options = new SequentialKNNOptions();

	/**
	 * Instantiates a new sequential knn builder.
	 * 
	 * @param config
	 *            the config
	 */
	public SequentialKNNBuilder(Configuration<SequentialKNNBuilder> config) {
		super(config);
		configure(config);
	}

	/**
	 * Instantiates a new sequential knn builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public SequentialKNNBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}
	
	/**
	 * instantiates a new SequentialKNNBuilder
	 * dimension and bias will need to be set manually using setters
	 */
	public SequentialKNNBuilder() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#build()
	 */
	@Override
	public SequentialKNN build() {
		SequentialKNN model = new SequentialKNN(getDimension(), bias)
				.setK(k)
				.setKDTreeType(kdType)
				.setLabelMizingType(mixing)
				.setSmoothertype(regType)
				.setSizeLimit(sizeLimit)
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
	protected SequentialKNNBuilder getThis() {
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#configure(com.
	 * parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<SequentialKNNBuilder> conf) {
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
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#getConfiguration()
	 */
	@Override
	public Configuration<SequentialKNNBuilder> getConfiguration() {
		Configuration<SequentialKNNBuilder> conf = new Configuration<SequentialKNNBuilder>(
				options);
		populateConfiguration(conf);
		return conf;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#populateConfiguration
	 * (com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<SequentialKNNBuilder> populateConfiguration(
			Configuration<SequentialKNNBuilder> conf) {
		super.populateConfiguration(conf);
		return conf;
	}

	/**
	 * Gets the options.
	 * 
	 * @return the options
	 */
	public static SequentialKNNOptions getOptions() {
		return new SequentialKNNOptions();
	}

	/**
	 * The Class SequentialKNNOptions.
	 */
	protected static class SequentialKNNOptions
			extends
			AbstractUpdateableKDTreeClassifierOptions<SequentialKNN, SequentialKNNBuilder> {

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions#
		 * getClassifierType()
		 */
		@Override
		public Classifiers getClassifierType() {
			return Classifiers.KNN;
		}
	}

}
