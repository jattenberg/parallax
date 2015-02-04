/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.rules;

import com.dsi.parallax.ml.classifier.AbstractClassifier;
import com.dsi.parallax.ml.classifier.ClassifierBuilder;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.option.Configuration;
import com.google.common.collect.Maps;

import java.util.Map;

// TODO: Auto-generated Javadoc
/**
 * most basic classifier, always predicts the mode label.
 * 
 * @author jattenberg
 */
public class ModeClassifier extends AbstractClassifier<ModeClassifier> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5124337353434200417L;

	/** The mode. */
	private double mode;

	/**
	 * Instantiates a new mode classifier.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public ModeClassifier(int dimension, boolean bias) {
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
		Map<Integer, Double> vals = Maps.newHashMap();
		double maxValue = Double.MIN_VALUE;
		for (Instance<BinaryClassificationTarget> inst : instances) {
			double label = inst.getLabel().getValue();
			int index = MLUtils.doubleHash(label);
			vals.put(index, 1 + (vals.containsKey(index) ? vals.get(index) : 0));

			if (vals.get(index) > maxValue) {
				maxValue = vals.get(index);
				mode = label;
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public ModeClassifier initialize() {
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
		return mode;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected ModeClassifier getModel() {
		return this;
	}

	/**
	 * The Class ModeClassifierBuilder.
	 */
	public static class ModeClassifierBuilder extends
			ClassifierBuilder<ModeClassifier, ModeClassifierBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -2931558039190986357L;

		/**
		 * Instantiates a new mode classifier builder.
		 * 
		 * @param config
		 *            the config
		 */
		public ModeClassifierBuilder(Configuration<ModeClassifierBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new mode classifier builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public ModeClassifierBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/**
		 * instantiates a new {@link ModeClassifierBuilder}
		 */
		public ModeClassifierBuilder() {
			super();
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public ModeClassifier build() {
			ModeClassifier model = new ModeClassifier(getDimension(), bias);
			model.setSmoothertype(regType)
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
		protected ModeClassifierBuilder getThis() {
			return this;
		}

		/**
		 * Gets the options.
		 * 
		 * @return the options
		 */
		public static ModeOptions getOptions() {
			return new ModeOptions();
		}

		/**
		 * The Class ModeOptions.
		 */
		public static class ModeOptions extends
				ClassifierOptions<ModeClassifier, ModeClassifierBuilder> {

			/*
			 * (non-Javadoc)
			 * 
			 * @see
			 * com.parallax.ml.classifier.ClassifierBuilder.ClassifierOptions
			 * #getClassifierType()
			 */
			@Override
			public Classifiers getClassifierType() {
				return Classifiers.MODE;
			}

		}
	}

}
