package com.parallax.ml.util.pair;

import java.util.Comparator;

public class FirstAscendingComparator implements Comparator<PrimitivePair>{
	/**
     * The method compares PrimitivePair 1 and PrimitivePair 2 and
     * sorts the pairs on the first field in ascending order
     * @param ex1 PrimitivePair
     * @param ex2 PrimitivePair
     * @return int
     */
	public int compare(PrimitivePair ex1, PrimitivePair ex2) {
		PrimitivePair l1 = (PrimitivePair) ex1;
		PrimitivePair l2 = (PrimitivePair) ex2;

		if (l1.first > l2.first)
			return -1;
		else if (l1.first < l2.first)
			return 1;
		else
			return 0;
	}
}
