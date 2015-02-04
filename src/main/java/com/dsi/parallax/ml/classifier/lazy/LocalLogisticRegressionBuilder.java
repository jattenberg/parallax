package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.OptionSet;
import org.apache.commons.cli.ParseException;

import java.lang.reflect.InvocationTargetException;

public class LocalLogisticRegressionBuilder
		extends
		AbstractUpdateableKDTreeClassifierBuilder<LocalLogisticRegression, LocalLogisticRegressionBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 7000465644943456538L;

	/** The options. */
	public static OptionSet<LocalLogisticRegressionBuilder> options = new LocalLogisticRegressionOptions();

	/**
	 * Instantiates a new sequential knn builder.
	 * 
	 * @param config
	 *            the config
	 */
	public LocalLogisticRegressionBuilder(
			Configuration<LocalLogisticRegressionBuilder> config) {
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
	public LocalLogisticRegressionBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}
	
	/**
	 * instantiates a new LocalLogisticRegressionBuilder
	 * dimension and bias will need ot be set manually using setters
	 */
	public LocalLogisticRegressionBuilder() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.ModelBuilder#build()
	 */
	@Override
	public LocalLogisticRegression build() {
		LocalLogisticRegression model = new LocalLogisticRegression(getDimension(),
				bias)
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
	protected LocalLogisticRegressionBuilder getThis() {
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
	public void configure(Configuration<LocalLogisticRegressionBuilder> conf) {
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
	public Configuration<LocalLogisticRegressionBuilder> getConfiguration() {
		Configuration<LocalLogisticRegressionBuilder> conf = new Configuration<LocalLogisticRegressionBuilder>(
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
	public Configuration<LocalLogisticRegressionBuilder> populateConfiguration(
			Configuration<LocalLogisticRegressionBuilder> conf) {
		super.populateConfiguration(conf);
		return conf;
	}

	/**
	 * Gets the options.
	 * 
	 * @return the options
	 */
	public static LocalLogisticRegressionOptions getOptions() {
		return new LocalLogisticRegressionOptions();
	}

	/**
	 * The Class LocalLogisticRegressionOptions.
	 */
	protected static class LocalLogisticRegressionOptions
			extends
			AbstractUpdateableKDTreeClassifierOptions<LocalLogisticRegression, LocalLogisticRegressionBuilder> {

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions#
		 * getClassifierType()
		 */
		@Override
		public Classifiers getClassifierType() {
			return Classifiers.LOCALLR;
		}
	}

	public static void main(String[] args) throws IllegalArgumentException,
			SecurityException, ParseException, IllegalAccessException,
			InvocationTargetException, NoSuchMethodException {
		ClassifierEvaluation.evaluate(args, LocalLogisticRegression.class);
	}
}
