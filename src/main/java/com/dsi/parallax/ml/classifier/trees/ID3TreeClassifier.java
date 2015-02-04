/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.trees;

import com.dsi.parallax.ml.classifier.ClassifierEvaluation;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.trees.*;
import com.dsi.parallax.ml.trees.Terminator;
import org.apache.commons.cli.ParseException;

import java.lang.reflect.InvocationTargetException;
import java.util.List;

// TODO: Auto-generated Javadoc
/**
 * The Class ID3TreeClassifier.
 */
public class ID3TreeClassifier extends AbstractTreeClassifier<ID3TreeClassifier> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5056614285411878557L;
	
	/** The builder. */
	private LogisticRegressionBuilder builder;

	/**
	 * Instantiates a new i d3 tree classifier.
	 *
	 * @param dimension the dimension
	 * @param bias the bias
	 */
	public ID3TreeClassifier(int dimension, boolean bias) {
		super(dimension, bias);
		initialize();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public ID3TreeClassifier initialize() {
		//TODO: replace with mode builder
		builder = new LogisticRegressionBuilder(dimension, bias);
		return model;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildCriterion()
	 */
	@Override
	protected SplitCriterion<BinaryClassificationTarget> buildCriterion() {
		return new BinaryTargetInfoGainSplitCriterion();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildSplitter()
	 */
	@Override
	protected Splitter<BinaryClassificationTarget> buildSplitter() {
		return new OrderedSplitter<BinaryClassificationTarget>();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildAttributeValueCache()
	 */
	@Override
	protected AttributeValueCache buildAttributeValueCache() {
		return new RecomputingAttributeValueCache();
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildLeafCreator()
	 */
	@Override
	protected LeafCreator<BinaryClassificationTarget> buildLeafCreator() {
		return new ClassifierLeafCreator(builder);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildPruner()
	 */
	@Override
	protected Pruner<BinaryClassificationTarget> buildPruner() {
		return null;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildProjectionFactory()
	 */
	@Override
	protected ProjectionFactory buildProjectionFactory() {
		return null;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildProjectionRatio()
	 */
	@Override
	protected double buildProjectionRatio() {
		return 1;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.classifier.trees.AbstractTreeClassifier#buildAdditionalTerminators()
	 */
	@Override
	protected List<Terminator<BinaryClassificationTarget>> buildAdditionalTerminators() {
		// no additional terminators needed for ID3
		return null;
	}
	
	/* (non-Javadoc)
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected ID3TreeClassifier getModel() {
		return this;
	}


	public static void main(String[] args) throws IllegalArgumentException, SecurityException, ParseException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
		ClassifierEvaluation.evaluate(args, ID3TreeClassifier.class);
	}
}
