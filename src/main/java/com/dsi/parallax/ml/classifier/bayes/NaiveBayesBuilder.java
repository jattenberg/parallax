package com.dsi.parallax.ml.classifier.bayes;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.UpdateableClassifierBuilder;
import com.dsi.parallax.ml.distributions.DistributionConfigurableBuilder;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.ConfigurableOption;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

// TODO: Auto-generated Javadoc
/**
 * The Class NaiveBayesBuilder.
 */
public class NaiveBayesBuilder extends
		UpdateableClassifierBuilder<NaiveBayes, NaiveBayesBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5666800373495648644L;

	/** The distribution builder. */
	private DistributionConfigurableBuilder distributionBuilder = new DistributionConfigurableBuilder();

	/** The prior distribution builder. */
	private DistributionConfigurableBuilder priorDistributionBuilder = new DistributionConfigurableBuilder();

	/** The document length normalization. */
	private double documentLengthNormalization = 100;

	/** The options. */
	public static OptionSet<NaiveBayesBuilder> options = new NaiveBayesOptions();

	/**
	 * Instantiates a new naive bayes builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public NaiveBayesBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/**
	 * Instantiates a new naive bayes builder. the dimension and bias will need
	 * to be set manually using setters
	 */
	public NaiveBayesBuilder() {
		super();
	}

	/**
	 * Instantiates a new naive bayes builder.
	 * 
	 * @param config
	 *            the configuration describing the desired model settings
	 */
	public NaiveBayesBuilder(Configuration<NaiveBayesBuilder> config) {
		super(config);
		configure(config);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#build()
	 */
	@Override
	public NaiveBayes build() {
		NaiveBayes out = new NaiveBayes(getDimension(), bias)
				.setDistributionBuilder(distributionBuilder)
				.setPriorDistributionBuilder(priorDistributionBuilder)
				.setDocumentLengthNormalization(documentLengthNormalization)
				.setSmoothertype(regType)
				.setPasses(passes)
				.setCrossvalidateSmootherTraining(crossValidateSmootherTraining)
				.initialize();
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#getThis()
	 */
	@Override
	protected NaiveBayesBuilder getThis() {
		return this;
	}

	/**
	 * Sets the document length normalization.
	 * 
	 * @param docLengthNormalization
	 *            the doc length normalization
	 * @return the naive bayes builder
	 */
	public NaiveBayesBuilder setDocumentLengthNormalization(
			double docLengthNormalization) {
		checkArgument(docLengthNormalization > 0,
				"docLengthNormalization must be positive. input: %s",
				docLengthNormalization);
		this.documentLengthNormalization = docLengthNormalization;
		return thisBuilder;
	}

	/**
	 * set the builder for liklihood distributions used in the naive bayes
	 * model.
	 * 
	 * @param distributionBuilder
	 *            builder for distirbutions
	 * @return the naive bayes builder
	 */
	public NaiveBayesBuilder setDistributionBuilder(
			DistributionConfigurableBuilder distributionBuilder) {
		this.distributionBuilder = distributionBuilder;
		return thisBuilder;
	}

	/**
	 * set the builder for likelihood distributions used in the naive bayes
	 * model using a configuraion.
	 * 
	 * @param distributionBuilderConfiguration
	 *            configuration for building a distribuiton builder
	 * @return the naive bayes builder
	 */
	public NaiveBayesBuilder setDistributionBuilder(
			Configuration<DistributionConfigurableBuilder> distributionBuilderConfiguration) {
		return setDistributionBuilder(new DistributionConfigurableBuilder(
				distributionBuilderConfiguration));
	}

	/**
	 * set the builder for prior distributions used in the naive bayes model.
	 * 
	 * @param priorDistributionBuilder
	 *            the prior distribution builder
	 * @return the naive bayes builder
	 */
	public NaiveBayesBuilder setPriorDistributionBuilder(
			DistributionConfigurableBuilder priorDistributionBuilder) {
		this.priorDistributionBuilder = priorDistributionBuilder;
		return thisBuilder;
	}

	/**
	 * set the builder for prior distributions used in the naive bayes model
	 * using a configuraion.
	 * 
	 * @param priorDistributionBuilderConfiguration
	 *            the prior distribution builder configuration
	 * @return the naive bayes builder
	 */
	public NaiveBayesBuilder setPriorDistributionBuilder(
			Configuration<DistributionConfigurableBuilder> priorDistributionBuilderConfiguration) {
		return setPriorDistributionBuilder(new DistributionConfigurableBuilder(
				priorDistributionBuilderConfiguration));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#configure(com.
	 * parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<NaiveBayesBuilder> conf) {
		super.configure(conf);
		setDocumentLengthNormalization(conf.floatOptionFromShortName("n"));
		@SuppressWarnings("unchecked")
		Configuration<DistributionConfigurableBuilder> priorBuilderConf = (Configuration<DistributionConfigurableBuilder>) conf
				.configurationFromShortName("P");
		@SuppressWarnings("unchecked")
		Configuration<DistributionConfigurableBuilder> likelihoodBuilderConf = (Configuration<DistributionConfigurableBuilder>) conf
				.configurationFromShortName("l");

		setPriorDistributionBuilder(priorBuilderConf);
		setDistributionBuilder(likelihoodBuilderConf);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifierBuilder#getConfiguration()
	 */
	@Override
	public Configuration<NaiveBayesBuilder> getConfiguration() {
		Configuration<NaiveBayesBuilder> conf = new Configuration<NaiveBayesBuilder>(
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
	public Configuration<NaiveBayesBuilder> populateConfiguration(
			Configuration<NaiveBayesBuilder> conf) {
		super.populateConfiguration(conf);
		conf.addFloatValueOnShortName("n", documentLengthNormalization);
		conf.addConfigurableValueOnShortName("P",
				priorDistributionBuilder.getConfiguration());
		conf.addConfigurableValueOnShortName("l",
				distributionBuilder.getConfiguration());
		return conf;
	}

	/**
	 * Gets the options.
	 * 
	 * @return the options
	 */
	public static NaiveBayesOptions getOptions() {
		return new NaiveBayesOptions();
	}

	/**
	 * The Class NaiveBayesOptions.
	 */
	protected static class NaiveBayesOptions extends
			UpdateableOptions<NaiveBayes, NaiveBayesBuilder> {
		{
			addOption(new FloatOption(
					"n",
					"doclengthnormalization",
					"Make the document have counts that sum to docLengthNormalization. I.e., if 20, it would be as if the document had 20 words.",
					100, false, new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(10000)));
			addOption(new ConfigurableOption<DistributionConfigurableBuilder>(
					"P", "prior", true,
					"builder for the prior distribution. options: "
							+ DistributionConfigurableBuilder.optionInfo(),
					new Configuration<DistributionConfigurableBuilder>(
							DistributionConfigurableBuilder.options)));

			addOption(new ConfigurableOption<DistributionConfigurableBuilder>(
					"l", "likelihood", true,
					"builder for the likelihood distributions. options: "
							+ DistributionConfigurableBuilder.optionInfo(),
					new Configuration<DistributionConfigurableBuilder>(
							DistributionConfigurableBuilder.options)));
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions#
		 * getClassifierType()
		 */
		@Override
		public Classifiers getClassifierType() {
			return Classifiers.NB;
		}
	}

}
