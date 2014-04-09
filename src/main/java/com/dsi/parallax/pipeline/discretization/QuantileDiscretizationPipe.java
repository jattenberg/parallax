/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.discretization;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.DiscreteDomains;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;
import com.google.common.collect.Ranges;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * map each continuous feature into an indicator for the value's qualtile
 * membership.
 *
 * @author jattenberg
 * 
 * TODO: extract method into own class
 */
public class QuantileDiscretizationPipe extends AbstractDiscretizationPipe {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8039936246483140194L;
	
	/** The size. */
	private int size;

	/**
	 * Instantiates a new quantile discretization pipe.
	 *
	 * @param expansionFactor the expansion factor
	 * @param keepContinuous the keep continuous
	 * @param readToInitialize the read to initialize
	 */
	public QuantileDiscretizationPipe(int expansionFactor,
			boolean keepContinuous, int readToInitialize) {
		super(expansionFactor, readToInitialize, keepContinuous);
	}

	/**
	 * Instantiates a new quantile discretization pipe.
	 *
	 * @param expansionFactor the expansion factor
	 * @param keepContinuous the keep continuous
	 * @param examinedFeatures the examined features
	 */
	public QuantileDiscretizationPipe(int expansionFactor,
			boolean keepContinuous, Set<Integer> examinedFeatures) {
		super(expansionFactor, keepContinuous, examinedFeatures);
	}

	/**
	 * Instantiates a new quantile discretization pipe.
	 *
	 * @param expansionFactor the expansion factor
	 * @param keepContinuous the keep continuous
	 */
	public QuantileDiscretizationPipe(int expansionFactor,
			boolean keepContinuous) {
		super(expansionFactor, keepContinuous);
	}

	/**
	 * Instantiates a new quantile discretization pipe.
	 *
	 * @param examinedFeatures the examined features
	 * @param expansionFactor the expansion factor
	 * @param keepContinuous the keep continuous
	 * @param readToInitialize the read to initialize
	 */
	public QuantileDiscretizationPipe(Set<Integer> examinedFeatures,
			int expansionFactor, boolean keepContinuous, int readToInitialize) {
		super(examinedFeatures, expansionFactor, keepContinuous,
				readToInitialize);
	}

	/**
	 * Adds the bounds info.
	 *
	 * @param dataCount the data count
	 * @param descriptiveStatistics the descriptive statistics
	 * @return the discrete bounds
	 */
	private DiscreteBounds addBoundsInfo(int dataCount,
			DescriptiveStatistics descriptiveStatistics) {
		double[] bounds = new double[expansionFactor];
		for (int i = 0; i < expansionFactor; i++)
			bounds[i] = descriptiveStatistics.getPercentile(100. * (i + 1.)
					/ (expansionFactor + .1));
		return new DiscreteBounds(size + dataCount*expansionFactor, bounds);
	}

	/**
	 * Adds the info.
	 *
	 * @param dataMap the data map
	 * @param vect the vect
	 */
	private void addInfo(Map<Integer, DescriptiveStatistics> dataMap,
			LinearVector vect) {
		for (int x_i : vect) {
			if(dataMap.keySet().contains(x_i))
				dataMap.get(x_i).addValue(vect.getValue(x_i));
		}

	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<QuantileDiscretizationPipe>() {
		}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractAccumulatingPipe#batchProcess(java.util.List)
	 */
	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		Map<Integer, DescriptiveStatistics> dataMap = Maps.newTreeMap();

		if (infoList.size() > 0) {
			LinearVector vect = infoList.get(0).getData();
			size = vect.size();
			
			for (int x_i : examinedFeatures == null ? Ranges
					.closedOpen(0, size).asSet(DiscreteDomains.integers())
					: examinedFeatures) {
				dataMap.put(x_i, new DescriptiveStatistics());
			}
			Iterator<LinearVector> vects = Iterators.transform(
					infoList.iterator(), uncontextifier);
			while (vects.hasNext()) {
				addInfo(dataMap, vects.next());
			}
			int ct = 0;
			for (int x_i : dataMap.keySet()) {
				DiscreteBounds bounds = addBoundsInfo(ct++, dataMap.get(x_i));
				indexToBounds.put(x_i, bounds);
			}
		}
		isTrained = true;
	}

}
