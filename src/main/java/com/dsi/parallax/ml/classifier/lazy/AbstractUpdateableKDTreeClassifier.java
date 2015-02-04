package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.classifier.AbstractUpdateableClassifier;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.KDTree;
import com.dsi.parallax.ml.util.KDTree.*;

import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

public abstract class AbstractUpdateableKDTreeClassifier<C extends AbstractUpdateableKDTreeClassifier<C>>
		extends AbstractUpdateableClassifier<C> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6879680807154117644L;

	public AbstractUpdateableKDTreeClassifier(int dimension, boolean bias) {
		super(dimension, false);
	}

	/** The kd tree. */
	protected KDTree<BinaryClassificationTarget> kdTree;

	/** The k. */
	protected int k = 3;

	/** The kd type. */
	protected KDType kdType = KDType.EUCLIDIAN;

	/** The mixing. */
	protected KNNMixingType mixing = KNNMixingType.MEAN;

	/** The size limit. */
	private int sizeLimit = Integer.MAX_VALUE;

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>> void updateModel(
			I inst) {
		// add point to tree
		kdTree.addPoint(inst, inst.getLabel());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(java.util.Collection
	 * )
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void updateModel(
			I instst) {
		for (Instance<BinaryClassificationTarget> inst : instst)
			updateModel(inst);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public C initialize() {
		kdTree = buildKDTree();
		return model;
	}

	/**
	 * Builds the kd tree.
	 * 
	 * @return the kD tree
	 */
	private KDTree<BinaryClassificationTarget> buildKDTree() {
		switch (kdType) {
		case EUCLIDIAN:
			return new SqrEuclid<BinaryClassificationTarget>(dimension,
					sizeLimit);
		case WEIGHTEDEUCLIDIAN:
			return new WeightedSqrEuclid<BinaryClassificationTarget>(dimension,
					sizeLimit);
		case MANHATTAN:
			return new Manhattan<BinaryClassificationTarget>(dimension,
					sizeLimit);
		case WEIGHTEDMANHATTAN:
			return new WeightedManhattan<BinaryClassificationTarget>(dimension,
					sizeLimit);
		default:
			throw new IllegalArgumentException(
					"entered an unrecognized kdType: " + kdType);
		}
	}

	protected List<Entry<BinaryClassificationTarget>> findNeighbors(
			Instance<?> inst) {
		return kdTree.nearestNeighbor(inst, k, false);
	}

	/**
	 * Gets the k.
	 * 
	 * @return the k
	 */
	public int getK() {
		return k;
	}

	/**
	 * Gets the kd type.
	 * 
	 * @return the kd type
	 */
	public KDType getKdType() {
		return kdType;
	}

	/**
	 * Gets the size limit.
	 * 
	 * @return the size limit
	 */
	public int getSizeLimit() {
		return sizeLimit;
	}

	/**
	 * Gets the mixing type.
	 * 
	 * @return the mixing type
	 */
	public KNNMixingType getMixingType() {
		return mixing;
	}

	/**
	 * Sets the kd tree type.
	 * 
	 * @param type
	 *            the type
	 * @return the sequential knn
	 */
	public C setKDTreeType(KDType type) {
		this.kdType = type;
		return model;
	}

	/**
	 * Sets the label mizing type.
	 * 
	 * @param mixingType
	 *            the mixing type
	 * @return the sequential knn
	 */
	public C setLabelMizingType(KNNMixingType mixingType) {
		this.mixing = mixingType;
		return model;
	}

	/**
	 * Sets the k.
	 * 
	 * @param k
	 *            the k
	 * @return the sequential knn
	 */
	public C setK(int k) {
		checkArgument(k >= 0, "k must be > 0 input: %s", k);
		this.k = k;
		return model;
	}

	/**
	 * Gets the mixing.
	 * 
	 * @return the mixing
	 */
	public KNNMixingType getMixing() {
		return mixing;
	}

	/**
	 * Sets the size limit.
	 * 
	 * @param sizeLimit
	 *            the size limit
	 * @return the sequential knn
	 */
	public C setSizeLimit(int sizeLimit) {
		checkArgument(sizeLimit >= 100, "sizeLimit must be >= 100, given: %s",
				sizeLimit);
		this.sizeLimit = sizeLimit;
		return model;
	}
}
