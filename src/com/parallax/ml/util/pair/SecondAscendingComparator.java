/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.pair;

import java.util.Comparator;

/**
 * SecondAscendingComparator sorts the pairs on the second field in increasing order
 *
 * @author Josh Attenberg
 */
public class SecondAscendingComparator implements Comparator<PrimitivePair> 
{
    /**
     * The method sorts the pairs on the second field in increasing order
     * @param ex1 first PrimitivePair
     * @param ex2 second PrimitivePair
     * @return int comparative result
     */
	public int compare(PrimitivePair ex1, PrimitivePair ex2) {
		PrimitivePair l1 = (PrimitivePair) ex1;
		PrimitivePair l2 = (PrimitivePair) ex2;

		if (l1.second > l2.second)
			return 1;
		else if (l1.second < l2.second)
			return -1;
		else
			return 0;
	}
}
