/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Set;

/**
 * matches a set of strings to positive and negative labels
 * 
 * @author jattenberg
 * 
 */
public class StringSetBinaryTargetParser implements
		TargetParser<BinaryClassificationTarget> {

	private final Set<String> positive, negative;
	private final boolean inferLabelOnMiss, missesArePositive;

	private StringSetBinaryTargetParser(Set<String> pos, Set<String> neg,
			boolean inferLabelOnMiss, boolean missesArePositive) {
		checkArgument((null != pos) || (null != neg),
				"positive and negative label sets are null! must specify some sets to match");
		positive = pos;
		negative = neg;
		this.inferLabelOnMiss = inferLabelOnMiss;
		this.missesArePositive = missesArePositive;

	}

	@Override
	public BinaryClassificationTarget parseTarget(String stringValue) {
		if (null == stringValue) {
			if (inferLabelOnMiss)
				return new BinaryClassificationTarget(missesArePositive ? 1 : 0);
			else
				return null;

		}
		if (null != positive) {
			if (positive.contains(stringValue))
				return new BinaryClassificationTarget(1);
		}
		if (null != negative) {
			if (negative.contains(stringValue))
				return new BinaryClassificationTarget(0);
		}
		if (inferLabelOnMiss)
			return new BinaryClassificationTarget(missesArePositive ? 1 : 0);
		else
			return null;
	}

	public static StringSetBinaryTargetParser buildStringSetBinaryTargetParser(
			Set<String> pos, Set<String> neg, boolean inferLabelOnMiss,
			boolean missesArePositive) {
		return new StringSetBinaryTargetParser(pos, neg, inferLabelOnMiss,
				missesArePositive);
	}

	public static StringSetBinaryTargetParser buildStringSetBinaryTargetParserNullMisses(
			Set<String> pos, Set<String> neg) {
		return new StringSetBinaryTargetParser(pos, neg, false, true);
	}

	public static StringSetBinaryTargetParser buildStringSetBinaryTargetParserPositiveMisses(
			Set<String> pos, Set<String> neg) {
		return new StringSetBinaryTargetParser(pos, neg, true, true);
	}
	
	public static StringSetBinaryTargetParser buildStringSetBinaryTargetParserNegativeMisses(
			Set<String> pos, Set<String> neg) {
		return new StringSetBinaryTargetParser(pos, neg, true, false);
	}

	public static StringSetBinaryTargetParser buildStringSetPositiveExamplesNullMisses(
			Set<String> pos) {
		return new StringSetBinaryTargetParser(pos, null, false, false);
	}

	public static StringSetBinaryTargetParser buildStringSetNegativeExamplesNullMisses(
			Set<String> neg) {
		return new StringSetBinaryTargetParser(null, neg, false, false);
	}
	
	public static StringSetBinaryTargetParser buildStringSetPositiveExamplesPosMiss(
			Set<String> pos) {
		return new StringSetBinaryTargetParser(pos, null, true, true);
	}

	public static StringSetBinaryTargetParser buildStringSetNegativeExamplesPosMiss(
			Set<String> neg) {
		return new StringSetBinaryTargetParser(null, neg, true, true);
	}
	
	public static StringSetBinaryTargetParser buildStringSetPositiveExamplesNegMiss(
			Set<String> pos) {
		return new StringSetBinaryTargetParser(pos, null, true, false);
	}

	public static StringSetBinaryTargetParser buildStringSetNegativeExamplesNegMiss(
			Set<String> neg) {
		return new StringSetBinaryTargetParser(null, neg, true, false);
	}
}
