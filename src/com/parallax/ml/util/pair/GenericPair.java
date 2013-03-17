/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.pair;

/**
 * GenericPair
 *
 * @param <T1>
 * @param <T2>
 * @author Josh Attenberg
 */
public class GenericPair<T1,T2>
{
	public final T1 first;
	public final T2 second;

    /**
     * Class constructor specifying T1 and T2 to create
     * @param first T1
     * @param second T2
     */
	public GenericPair(T1 first, T2 second)
	{
		this.first=first;
		this.second=second;
	}

    /**
     * The method gets the first generic object
     * @return T1
     */
	public T1 getFirst()
	{
		return first;
	}

    /**
     * The method gets the second generic object
     * @return T2
     */
	public T2 getSecond()
	{
		return second;
	}

	@Override
	public int hashCode()
	{
		return 37*first.hashCode()+second.hashCode();
	}
	
	@Override
	public boolean equals(Object o)
	{
		if(! (o instanceof GenericPair<?, ?>))
			return false;

		@SuppressWarnings("rawtypes")
        GenericPair gp = (GenericPair)o;
		
		return gp.first.equals(first)&&gp.second.equals(second);
		
	}
	
}
