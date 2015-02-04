/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.common.collect.Sets;

import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * from BenHaim and Tom-Tov 2010 stores a histogram as a set of pairs consisting
 * of pairs {p,m} where p is a point (numerical value) and m is a count (items
 * in the bin)
 * 
 * @author jattenberg
 * 
 */
public class SequentialHistogram {
	private SortedMap<Double, Double> hist;
	int bins; // bins in the histogram;
	double tot;

	public SequentialHistogram(int bins) {
		hist = new TreeMap<Double, Double>();
		this.bins = bins;
		tot = 0;
	}

	public void add(double p) {
		hist.put(p, hist.containsKey(p) ? 1 + hist.get(p) : 1);
		tot++;
		if (hist.size() > bins)
			cleanUp();
	}

	private void cleanUp() {
		double maxDiff = Double.MIN_VALUE;
		double maxKey = hist.firstKey();
		double secondMaxKey = Double.MIN_VALUE;
		double lastKey = hist.firstKey();

		for (double key : Sets.difference(hist.keySet(),
				Sets.newHashSet(hist.firstKey()))) {
			double diff = key - lastKey;
			if (diff > maxDiff) {
				maxDiff = diff;
				maxKey = lastKey;
				secondMaxKey = key;
			}
			lastKey = key;
		}
		double val1 = hist.remove(maxKey);
		double val2 = hist.remove(secondMaxKey);

		hist.put((maxKey * val1 + secondMaxKey * val2) / (val1 + val2), val1
				+ val2);
	}

	public void merge(SequentialHistogram toMerge) {
		for (double key : toMerge.hist.keySet()) {
			double value = toMerge.hist.get(key);
			hist.put(key, hist.containsKey(key) ? value + hist.get(key) : value);
			tot += value;
		}

		while (hist.size() > bins)
			cleanUp();
	}

	public double closestBin(double query) {
		if (hist.containsKey(query))
			return query;
		TreeSet<Double> keys = Sets.newTreeSet(hist.keySet());
		PeekingIterator<Double> aboveIterator, belowIterator;
		belowIterator = Iterators.peekingIterator(keys.headSet(query, false)
				.descendingIterator());
		aboveIterator = Iterators.peekingIterator(keys.tailSet(query, false)
				.iterator());

		if (!belowIterator.hasNext()) {
			if (!aboveIterator.hasNext())
				throw new IllegalStateException(
						"attempting to query an empty histogram");
			else
				return aboveIterator.next();
		} else if (!aboveIterator.hasNext()) {
			return belowIterator.next();
		} else {

			double prior = belowIterator.next();
			double next = aboveIterator.next();

			double priorDistance = query - prior;
			double nextDistance = next - query;

			if (priorDistance < nextDistance)
				return prior;
			else
				return next;
		}
	}

	public double probability(double query) {
		double closest = closestBin(query);
		return hist.get(closest) / tot;
	}

	// TODO: edge cases?
	public double sum(double query) {
		TreeSet<Double> keys = Sets.newTreeSet(hist.keySet());

		PeekingIterator<Double> aboveIterator, belowIterator;
		belowIterator = Iterators.peekingIterator(keys.headSet(query, false)
				.descendingIterator());
		aboveIterator = Iterators.peekingIterator(keys.tailSet(query, false)
				.iterator());

		if (!belowIterator.hasNext())
			return 0;

		double prior = belowIterator.next();
		double next = aboveIterator.next();
		double priorVal = hist.get(prior);
		double nextVal = hist.get(next);

		double m_b = priorVal + (query - prior) * (nextVal - priorVal)
				/ (next - prior);
		double s = (priorVal + m_b) * (query - prior)
				/ (2. * next - 2. * prior);

		while (belowIterator.hasNext()) {
			double key = belowIterator.next();
			double value = hist.get(key);
			s += value;
		}
		s += prior / 2.;
		return s;
	}
}
