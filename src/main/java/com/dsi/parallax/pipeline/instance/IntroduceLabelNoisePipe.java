/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import static com.google.common.base.Preconditions.checkArgument;

import java.lang.reflect.Type;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

/**
 * A pipe that randomly flips the binary labels of incoming instances according
 * to a specified probability
 */
public class IntroduceLabelNoisePipe
		extends
		AbstractPipe<BinaryClassificationInstance, BinaryClassificationInstance> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -4841403232733748942L;

	/** The noise ratio; probability that a label is flipped. */
	private final double noiseRatio;

	/**
	 * Instantiates a new introduce label noise pipe.
	 * 
	 * @param noiseRatio
	 *            he noise ratio; probability that a label is flipped
	 */
	public IntroduceLabelNoisePipe(double noiseRatio) {
		super();
		checkArgument(noiseRatio >= 0 && noiseRatio <= 1,
				"noise ratio must be between 0 and 1, input %s", noiseRatio);
		this.noiseRatio = noiseRatio;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<IntroduceLabelNoisePipe>() {
		}.getType();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<BinaryClassificationInstance> operate(
			Context<BinaryClassificationInstance> context) {
		if (MLUtils.GENERATOR.nextDouble() < noiseRatio) {
			double label = context.getData().getLabel().getValue();
			label = 1 - label;
			context.getData().setLabel(new BinaryClassificationTarget(label));

		}
		return context;

	}

}
