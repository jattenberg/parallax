/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.lazy;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.KDTree.Entry;
import com.google.common.collect.Maps;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import java.util.List;
import java.util.Map;

// TODO: Auto-generated Javadoc
/**
 * The Enum KNNMixingType.
 */
public enum KNNMixingType {

	/** The mean. */
	MEAN {
		DescriptiveStatistics stats = new DescriptiveStatistics();

		@Override
		public double computeScore(

		List<Entry<BinaryClassificationTarget>> neighborLabels) {
			stats.clear();
			for (Entry<BinaryClassificationTarget> e : neighborLabels)
				stats.addValue(e.value.getValue());
			return stats.getMean();
		}
	},
	
	/** The median. */
	MEDIAN {
		DescriptiveStatistics stats = new DescriptiveStatistics();

		@Override
		public double computeScore(
				List<Entry<BinaryClassificationTarget>> neighborLabels) {
			stats.clear();
			for (Entry<BinaryClassificationTarget> e : neighborLabels)
				stats.addValue(e.value.getValue());
			return stats.getPercentile(50);
		}
	},
	
	/** The mode. */
	MODE {
		@Override
		public double computeScore(
				List<Entry<BinaryClassificationTarget>> neighborLabels) {
			Map<Double, Double> vals = Maps.newHashMap();
			double maxValue = Double.MIN_VALUE;
			double max = 0;

			for (Entry<BinaryClassificationTarget> e : neighborLabels) {
				double score = e.value.getValue();
				vals.put(score,
						(vals.containsKey(score) ? vals.get(score) + 1 : 1));
				if (vals.get(score) > maxValue) {
					maxValue = vals.get(score);
					max = score;
				}
			}
			return max;
		}
	};

	/**
	 * Compute score.
	 *
	 * @param neighborLabels the neighbor labels
	 * @return the double
	 */
	public abstract double computeScore(
			List<Entry<BinaryClassificationTarget>> neighborLabels);
}
