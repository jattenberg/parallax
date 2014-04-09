/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.List;
import java.util.Queue;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PegasosBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;
import com.google.common.collect.Lists;

// TODO: Auto-generated Javadoc
/**
 * pegasos algorithm for solving regularized SVM see: Pegasos: Primal Estimated
 * sub-GrAdient SOlver for SVM TODO: make update efficient by estimating changes
 * in norms.
 * 
 * @author josh
 */
public class Pegasos extends AbstractLinearUpdateableClassifier<Pegasos> {

	/** The k. */
	private int windowSize = 1;

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5466274548530020022L;

	/** The cache. */
	private Queue<Instance<BinaryClassificationTarget>> cache;

	/** The trained. */
	private boolean trained = false;

	private static final AnnealingSchedule SCHEDULE = new ConstantAnnealingSchedule(
			1d);

	/**
	 * Instantiates a new pegasos.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public Pegasos(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.classifier.linearupdateable.
	 * AbstractLinearUpdateableClassifier
	 * #computeUpdateGradient(com.parallax.ml.instance.Instanze)
	 */
	@Override
	protected <I extends Instance<BinaryClassificationTarget>> Gradient computeUpdateGradient(
			I inst) {
		cache.add(inst);
		trained = false;

		if (cache.size() >= windowSize) {
			return pegasosTrain();
		} else {
			return new Gradient(LinearVectorFactory.getVector(dimension));
		}
	}

	/**
	 * Pegasos train.
	 */
	private Gradient pegasosTrain() {
		List<Instance<BinaryClassificationTarget>> aplus = Lists
				.newLinkedList();
		for (Instance<BinaryClassificationTarget> x : cache) {
			double label = x.getLabel().getValue();
			double innerproduct = innerProduct(x);
			double err = MLUtils.probToSVMInterval(label) * innerproduct;
			if (err < 1) {
				aplus.add(x);
			}
		}
		LinearVector gradientVector = LinearVectorFactory.getVector(dimension);

		if (aplus.size() != 0) {

			double eta = 1. / (regularizationWeight * (epoch + 1));

			LinearVector wHalf = LinearVectorFactory.getVector(dimension)
					.plusVectorTimes(vec, (1d - eta * regularizationWeight));

			for (Instance<BinaryClassificationTarget> x : aplus) {
				double slab = (eta / (double) aplus.size())
						* MLUtils.probToSVMInterval(x.getLabel().getValue()) * x.getWeight();
				for (int x_i : x) {
					double update = slab * x.getFeatureValue(x_i);
					wHalf.updateValue(x_i, update);
				}
				if (bias) {
					wHalf.updateValue(dimension - 1, slab);
				}
			}
			double wnorm = wHalf.L2Norm();

			double norm = Math.min(1., (1 / Math.sqrt(regularizationWeight))
					/ Math.sqrt(wnorm));
			wHalf.timesEquals(norm);
			gradientVector = wHalf.minus(vec);
		}

		cache.clear();
		trained = true;

		return new Gradient(gradientVector);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public Pegasos initialize() {
		initW(1 / Math.sqrt(regularizationWeight));
		cache = Lists.newLinkedList();
		return this;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected Pegasos getModel() {
		return this;
	}

	/**
	 * Sets the window size.
	 * 
	 * @param windowsize
	 *            the windowsize
	 * @return the pegasos
	 */
	public Pegasos setWindowSize(int windowsize) {
		checkArgument(windowsize >= 1, "window size must be > 1. input: %s",
				windowsize);
		this.windowSize = windowsize;
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
	public double regress(Instance<?> x) {
		if (!trained) {
			pegasosTrain();
		}
		return innerProduct(x);
	}

	/**
	 * Gets the window size.
	 * 
	 * @return the window size
	 */
	public int getWindowSize() {
		return windowSize;
	}

	@Override
	public AnnealingSchedule getAnnealingSchedule() {
		return SCHEDULE;
	}

	/**
	 * Builder.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 * @return the pegasos builder
	 */
	public PegasosBuilder builder(int dimension, boolean bias) {
		return new PegasosBuilder(dimension, bias);
	}

	/**
	 * The main method.
	 * 
	 * @param args
	 *            the arguments
	 * @throws Exception
	 *             the exception
	 */
	public static void main(String[] args) throws Exception {
		ClassifierEvaluation.evaluate(args, Pegasos.class);
	}

}
