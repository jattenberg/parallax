package com.dsi.parallax.ml.classifier.smoother;

import java.util.List;

import com.dsi.parallax.ml.util.KDTree;
import com.dsi.parallax.ml.util.KDTree.Entry;
import com.dsi.parallax.ml.util.KDTree.SqrEuclid;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;

import static com.google.common.base.Preconditions.checkArgument;

public abstract class AbstractKNNSmoother<R extends AbstractKNNSmoother<R>>
		extends AbstractUpdateableSmoother<R> {

	private static final long serialVersionUID = 7829567875055513579L;

	/**
	 * The Constant LIMIT. (max size of the KD Tree, and the Constant DEFAULTK
	 * (consider 50 neighbors for KNN as a default)
	 */
	protected static final int DEFAULTK = 50, LIMIT = 10000;

	/** The k used for the KNN algorithm. */
	protected int k;

	/** The tree. */
	protected final KDTree<Double> tree;

	public final R thisSmoother;

	public AbstractKNNSmoother() {
		this(DEFAULTK);
	}

	public AbstractKNNSmoother(int k) {
		this.k = k;
		this.tree = new SqrEuclid<Double>(1, LIMIT);
		thisSmoother = getSmoother();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.UpdateableSmoother#update(com
	 * .parallax.ml.util.pair.PrimitivePair)
	 */
	@Override
	public void update(PrimitivePair p) {
		tree.addPoint(pointToVector(p.first), p.second);

	}

	/**
	 * Instantiates a new kNN smoother.
	 */

	protected List<Entry<Double>> findNeighbors(double prediction) {
		return tree.nearestNeighbor(pointToVector(prediction),
				(int) Math.min(k, tree.size()), false);
	}

	public abstract R getSmoother();

	public R setK(int k) {
		checkArgument(k > 0, "k must be positive, given: %s", k);
		this.k = k;
		return thisSmoother;
	}

	public int getK() {
		return k;
	}

	protected static LinearVector pointToVector(double point) {
		return LinearVectorFactory.getDenseVector(new double[] { point });
	}
}
