package com.parallax.ml.classifier.trees;

import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.util.option.Configuration;

/**
 * Builder for {@link ID3TreeClassifier}s.
 */
public class ID3Builder extends
		TreeClassifierBuilder<ID3TreeClassifier, ID3Builder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1965080464398082206L;

	/**
	 * Instantiates a new i d3 builder.
	 * 
	 * @param config
	 *            containing the desited settings for the id3 classifier
	 */
	public ID3Builder(Configuration<ID3Builder> config) {
		super(config);
		configure(config);
	}

	/**
	 * instantiates a new ID3Builder dimension and bias must be set manually
	 * using setters
	 */
	public ID3Builder() {
		super();
	}

	/**
	 * Instantiates a new i d3 builder.
	 * 
	 * @param dimension
	 *            the number of features in the instantiated classifier
	 * @param bias
	 *            should the model have an additional (+1) intercept term?
	 */
	public ID3Builder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#build()
	 */
	@Override
	public ID3TreeClassifier build() {
		ID3TreeClassifier model = new ID3TreeClassifier(getDimension(), bias);
		model.setMaxDepth(maxDepth)
				.setMinEntropy(minEntropy)
				.setMinExamples(minExamples)
				.setPrepruningAttempts(prepruningAttempts)
				.setProjectionRatio(projectionRatio)
				.setSmoothertype(regType)
				.setCrossvalidateSmootherTraining(crossValidateSmootherTraining)
				.initialize();
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#getThis()
	 */
	@Override
	protected ID3Builder getThis() {
		return this;
	}

	/**
	 * configures the settings of the id3 buidler from the given configuration <br>
	 * ugly. fuck my life.
	 * 
	 * @param conf
	 *            containing the desired settings of the id3 model
	 */
	@Override
	public void configure(Configuration<ID3Builder> conf) {
		super.configure(conf);
		// Configuration<? extends ClassifierBuilder<?, ?>> classifierConfig =
		// (Configuration<? extends ClassifierBuilder<?, ?>>) conf
		// .configurationFromShortName("c");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.trees.TreeClassifierBuilder#getConfiguration()
	 */
	@Override
	public Configuration<ID3Builder> getConfiguration() {
		Configuration<ID3Builder> conf = new Configuration<ID3Builder>(
				new ID3TreeOptions());
		return populateConfiguration(conf);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.trees.TreeClassifierBuilder#populateConfiguration
	 * (com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<ID3Builder> populateConfiguration(
			Configuration<ID3Builder> conf) {
		super.populateConfiguration(conf);
		// conf.addConfigurableValueOnShortName("c", new MeanClassifierBuilder(
		// dimension, bias).getConfiguration());
		return conf;
	}

	/**
	 * ID3TreeOptions for configuring ID3 builders the only option is c /
	 * classifierBuilder [configuration] - nested configuration of a classifier
	 * builder used to train leaf nodes
	 */
	protected class ID3TreeOptions extends
			TreeClassifierOptions<ID3TreeClassifier, ID3Builder> {
		{

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions#
		 * getClassifierType()
		 */
		@Override
		public Classifiers getClassifierType() {
			return Classifiers.ID3;
		}
	}
}
