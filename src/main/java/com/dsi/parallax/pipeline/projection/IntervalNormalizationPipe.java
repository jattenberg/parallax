package com.dsi.parallax.pipeline.projection;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.pipeline.AbstractAccumulatingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Collections2;
import com.google.common.collect.Maps;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.math.stat.descriptive.SummaryStatistics;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public class IntervalNormalizationPipe extends
		AbstractAccumulatingPipe<LinearVector, LinearVector> {

	private static final long serialVersionUID = -9173101330232774227L;
	private static final double DEFAULT_MIN = 0d, DEFAULT_MAX = 1d;
	private final double min, max;

	Map<Integer, PrimitivePair> minMaxDimPairs;

	public IntervalNormalizationPipe() {
		this(DEFAULT_MIN, DEFAULT_MAX);
	}

	public IntervalNormalizationPipe(double min, double max) {
		this(min, max, -1);
	}

	public IntervalNormalizationPipe(int toConsider) {
		this(DEFAULT_MIN, DEFAULT_MAX, toConsider);
	}

	public IntervalNormalizationPipe(double min, double max, int toConsider) {
		super(toConsider);
		this.min = min;
		this.max = max;
	}

	@Override
	public Type getType() {
		return new TypeToken<IntervalNormalizationPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		return Context.createContext(context, minMaxProject(context.getData()));

	}

	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		Map<Integer, SummaryStatistics> stats = Maps.newHashMap();
		minMaxDimPairs = Maps.newHashMap();

		for (LinearVector vect : Collections2.transform(infoList,
				uncontextifier)) {
			for (int x_i : vect) {
				if (!stats.containsKey(x_i)) {
					stats.put(x_i, new SummaryStatistics());
				}
				stats.get(x_i).addValue(vect.getValue(x_i));
			}
		}
		for (int x_i : stats.keySet()) {
			double min = stats.get(x_i).getMin();
			double max = stats.get(x_i).getMax();
			minMaxDimPairs.put(x_i, new PrimitivePair(min, max));
		}
		isTrained = true;
	}

	private LinearVector minMaxProject(LinearVector input) {
		LinearVector output = LinearVectorFactory.getVector(input.size());

		for (int x_i : input) {
			double value = input.getValue(x_i);
			if (!minMaxDimPairs.containsKey(x_i)) {
				output.resetValue(x_i, value);
			} else {
				PrimitivePair minMaxDim = minMaxDimPairs.get(x_i);
				double dimMin = minMaxDim.first, dimMax = minMaxDim.second;

				double score = ((value - dimMin) / (dimMax - dimMin))
						* (max - min) + min;
				output.resetValue(x_i, MLUtils.floatingPointEquals(
						(dimMax - dimMin), 0) ? value : score);
			}
		}
		return output;
	}

}
