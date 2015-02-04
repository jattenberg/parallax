/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import com.dsi.parallax.ml.util.MLUtils;
import org.apache.log4j.Logger;

/**
 * BinaryTargetNumericParser converts the string label into a BinaryClassificationTarget
 * 
 * @author jattenberg
 *
 */
public class BinaryTargetNumericParser implements
		TargetParser<BinaryClassificationTarget> {

	private static Logger LOGGER = Logger
			.getLogger(BinaryTargetNumericParser.class);

	@Override
	public BinaryClassificationTarget parseTarget(String stringValue) {
		if (checkValidBinaryNumericLabel(stringValue))
			return new BinaryClassificationTarget(
					Double.parseDouble(stringValue));
		else {
			LOGGER.info(stringValue
					+ " is not a valid numeric target. Looking for numbers between 0 and 1");
			return null;
		}
	}

	private static boolean checkValidBinaryNumericLabel(String label) {
		if (label == null)
			return false;
		if (!MLUtils.isNumeric(label)) {
			return false;
		}
		double d = Double.parseDouble(label);
		if (d >= 0 && d <= 1)
			return true;
		else
			return false;
	}
}
