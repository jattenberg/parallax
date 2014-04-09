/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;

// TODO: Auto-generated Javadoc
/**
 * The Class HingeLoss.
 */
public class HingeLoss {
	
	/** The thresh. */
	private final double thresh;

	/**
	 * thresh is the margin.
	 *
	 * @param thresh the thresh
	 */
	public HingeLoss(double thresh) {
		this.thresh = thresh;
	}

	/**
	 * Loss.
	 *
	 * @param label the label
	 * @param prediction the prediction
	 * @return the double
	 */
	public double loss(BinaryClassificationTarget label,
			BinaryClassificationTarget prediction) {
		return loss(label.getValue(), prediction.getValue());
	}

	/**
	 * Loss.
	 *
	 * @param label the label
	 * @param prediction the prediction
	 * @return the double
	 */
	public double loss(double label, double prediction) {
		return Math.max(
				0,
				thresh - MLUtils.probToSVMInterval(label)
						* MLUtils.probToSVMInterval(prediction));
	}
}
