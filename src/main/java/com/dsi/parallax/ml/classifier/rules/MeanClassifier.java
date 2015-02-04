/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.rules;

import com.dsi.parallax.ml.classifier.AbstractClassifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.rules.MeanClassifier.MeanClassifierBuilder.MeanOptions;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.OptionSet;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

// TODO: Auto-generated Javadoc
/**
 * The Class MeanClassifier.
 */
public class MeanClassifier extends AbstractClassifier<MeanClassifier> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8063082830916446713L;

	/** The mean. */
	private double mean;

	private static OptionSet<MeanClassifierBuilder> options = new MeanOptions();

	/**
	 * Instantiates a new mean classifier.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public MeanClassifier(int dimension, boolean bias) {
		super(dimension, bias);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#train(com.parallax.ml.instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void modelTrain(
			I instances) {
		DescriptiveStatistics stats = new DescriptiveStatistics();
		for (Instance<BinaryClassificationTarget> inst : instances) {
			stats.addValue(inst.getLabel().getValue());
		}
		mean = stats.getMean();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public MeanClassifier initialize() {
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	protected double regress(Instance<?> inst) {
		return mean;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected MeanClassifier getModel() {
		return this;
	}

	/**
	 * The Class MeanClassifierBuilder.
	 */
	public static class MeanClassifierBuilder extends
			ClassifierBuilder<MeanClassifier, MeanClassifierBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -2931558039190986357L;

		/**
		 * Instantiates a new mean classifier builder.
		 * 
		 * @param config
		 *            the config
		 */
		public MeanClassifierBuilder(Configuration<MeanClassifierBuilder> config) {
			super(config);
			configure(config);
		}
		
		/**
		 * instantiates a new {@link MeanClassifierBuilder}
		 */
		public MeanClassifierBuilder() {
			super();
		}

		/**
		 * Instantiates a new mean classifier builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public MeanClassifierBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public MeanClassifier build() {
			MeanClassifier model = new MeanClassifier(getDimension(), bias);
			model.setSmoothertype(regType)
					.setCrossvalidateSmootherTraining(
							crossValidateSmootherTraining).initialize();
			return model;

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * com.parallax.ml.classifier.UpdateableClassifierBuilder#getConfiguration
		 * ()
		 */
		@Override
		public Configuration<MeanClassifierBuilder> getConfiguration() {
			Configuration<MeanClassifierBuilder> conf = new Configuration<MeanClassifierBuilder>(
					options);
			populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.UpdateableClassifierBuilder#
		 * populateConfiguration (com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<MeanClassifierBuilder> populateConfiguration(
				Configuration<MeanClassifierBuilder> conf) {
			super.populateConfiguration(conf);
			return conf;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected MeanClassifierBuilder getThis() {
			return this;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static MeanOptions getOptions() {
			return new MeanOptions();
		}

		/**
		 * The Class MeanOptions.
		 */
		protected static class MeanOptions extends
				ClassifierOptions<MeanClassifier, MeanClassifierBuilder> {

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.MEAN;
			}

		}
	}
}
