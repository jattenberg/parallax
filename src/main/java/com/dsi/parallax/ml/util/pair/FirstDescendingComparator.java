/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.pair;

import java.util.Comparator;

/**
 * FirstDescendingComparator sorts the pairs on the first field in decreasing order
 *
 * @author Josh Attenberg
 */
public class FirstDescendingComparator implements Comparator<PrimitivePair>
{
	/**
     * The method compares PrimitivePair 1 and PrimitivePair 2 and
     * sorts the pairs on the first field in decreasing order
     * @param ex1 PrimitivePair
     * @param ex2 PrimitivePair
     * @return int
     */
	@Override
	public int compare(PrimitivePair ex1, PrimitivePair ex2) {
		PrimitivePair l1 = ex1;
		PrimitivePair l2 = ex2;

		if (l1.first > l2.first)
			return 1;
		else if (l1.first < l2.first)
			return -1;
		else
			return 0;
	}
}
