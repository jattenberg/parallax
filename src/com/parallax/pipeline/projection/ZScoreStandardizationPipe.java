package com.parallax.pipeline.projection;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

import org.apache.commons.math.stat.descriptive.SummaryStatistics;

import com.google.common.collect.Collections2;
import com.google.common.collect.Maps;
import com.google.gson.reflect.TypeToken;
import com.parallax.ml.util.MLUtils;
import com.parallax.ml.util.pair.PrimitivePair;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;
import com.parallax.pipeline.AbstractAccumulatingPipe;
import com.parallax.pipeline.Context;

public class ZScoreStandardizationPipe extends
		AbstractAccumulatingPipe<LinearVector, LinearVector> {

	private static final long serialVersionUID = 522307374139407166L;

	private Map<Integer, PrimitivePair> meanStdDevPairs;

	public ZScoreStandardizationPipe() {
		this(-1);
	}

	public ZScoreStandardizationPipe(int toConsider) {
		super(toConsider);
		meanStdDevPairs = Maps.newHashMap();
	}

	@Override
	public Type getType() {
		return new TypeToken<ZScoreStandardizationPipe>() {
		}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<LinearVector> context) {
		return Context.createContext(context, ZScoreProject(context.getData()));
	}

	@Override
	protected void batchProcess(List<Context<LinearVector>> infoList) {
		Map<Integer, SummaryStatistics> stats = Maps.newHashMap();
		for (LinearVector vect : Collections2.transform(infoList,
				uncontextifier)) {
			for (int x_i = 0; x_i < vect.size(); x_i++) {
				if (!stats.containsKey(x_i)) {
					stats.put(x_i, new SummaryStatistics());
				}
				stats.get(x_i).addValue(vect.getValue(x_i));
			}
		}

		for (int x_i : stats.keySet()) {
			double mean = stats.get(x_i).getMean();
			double stdDev = stats.get(x_i).getStandardDeviation();
			meanStdDevPairs.put(x_i, new PrimitivePair(mean, stdDev));
		}
		isTrained = true;
	}

	private LinearVector ZScoreProject(LinearVector input) {
		LinearVector output = LinearVectorFactory.getVector(input.size());

		for (int x_i = 0; x_i < output.size(); x_i++) {
			double value = output.getValue(x_i);
			if (!meanStdDevPairs.containsKey(x_i)) {
				output.resetValue(x_i, value);
			} else {
				PrimitivePair meanStdDevPair = meanStdDevPairs.get(x_i);
				double zscore = (value - meanStdDevPair.first)
						/ meanStdDevPair.second;
				output.resetValue(x_i, MLUtils.floatingPointEquals(
						meanStdDevPair.second, 0) ? value : zscore);
			}
		}
		return output;
	}
}
