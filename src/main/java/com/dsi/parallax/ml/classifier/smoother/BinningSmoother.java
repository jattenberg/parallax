package com.dsi.parallax.ml.classifier.smoother;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.dsi.parallax.ml.util.pair.FirstDescendingComparator;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.google.common.collect.Lists;

// TODO: Auto-generated Javadoc
/**
 * BinningSmoother. consists of sorting the examples in decreasing order by
 * their estimated probabilities and dividing the set into k bins (i.e., subsets
 * of equal size).
 * 
 * See:
 * {@link <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.3039&rep=rep1&type=pdf">Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers</a>}
 * For more info
 */
public class BinningSmoother extends AbstractSmoother<BinningSmoother> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -1868667691923768094L;

	/** The Constant DEFAULTBINS. */
	private static final int DEFAULTBINS = 10;

	/** The bins used to divide the range of incoming scores. */
	private final int bins;

	/** The indicies. */
	private List<Double> indicies;

	/** The values. */
	private List<Double> values;

	/**
	 * Instantiates a new binning smoother.
	 */
	public BinningSmoother() {
		this(DEFAULTBINS);
	}

	/**
	 * Instantiates a new binning smoother.
	 * 
	 * @param bins
	 *            the number of bins used to divide the range of input scores
	 */
	public BinningSmoother(int bins) {
		checkArgument(bins > 0, "bins must be positive, given: %s", bins);
		this.bins = bins;
		indicies = Lists.newArrayList();
		values = Lists.newArrayList();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#smooth(double)
	 */
	@Override
	public double smooth(double prediction) {
		int index = findIndex(prediction);
		return values.get(index);
	}

	/**
	 * Find index of the bin to compare for the input prediction.
	 * 
	 * @param prediction
	 *            the prediction to look up
	 * @return the index of the bin containing the given prediction
	 */
	protected int findIndex(double prediction) {
		int index = Collections.binarySearch(indicies, prediction);
		if (index < 0) {
			index = -index - 1;
		}
		return index == indicies.size() ? index - 1 : index;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.smoother.Smoother#train(java.util.Collection
	 * )
	 */
	@Override
	public void train(Collection<PrimitivePair> input) {
		// 1. sort the incoming pairs in increasing order:
		List<PrimitivePair> ordered = Lists.newArrayList(input);
		Collections.sort(ordered, new FirstDescendingComparator());

		int usedBins = (int) Math.min(bins, ordered.size());
		int binSize = ordered.size() / usedBins;

		for (int bin = 0; bin < usedBins; bin++) {
			int start = bin * binSize;
			int end = (int) Math.min(start + binSize, ordered.size());

			double max = ordered.get(end - 1).first;
			double mean = listMean(ordered.subList(start, end));

			indicies.add(max);
			values.add(mean);
		}
	}

	/**
	 * List mean.
	 * 
	 * @param subList
	 *            the sub list
	 * @return the double
	 */
	private double listMean(List<PrimitivePair> subList) {
		if (subList.size() == 0) {
			return 0;
		} else {
			double sum = 0;
			for (PrimitivePair pair : subList) {
				sum += pair.second;
			}
			return sum / subList.size();
		}
	}

}
