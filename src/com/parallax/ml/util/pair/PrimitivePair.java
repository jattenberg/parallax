/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.pair;

import com.parallax.ml.util.MLUtils;

/**
 * PrimitivePair is JavaBean
 *
 * @author Josh Attenberg
 */
public class PrimitivePair
{
	public double first;
	public double second;

    /**
     * Class constructor specifying the first and second number to create
     * @param first first number
     * @param second second number
     */
	public PrimitivePair(double first, double second)
	{
		this.first = first;
		this.second = second;
	}

    /**
     * The method gets first number
     * @return first number
     */
	public double getFirst() {
		return first;
	}

    /**
     * The method sets first number
     * @param fisrt first number
     */
	public void setFirst(double fisrt) {
		this.first = fisrt;
	}

    /**
     * The method gets second number
     * @return second number
     */
	public double getSecond() {
		return second;
	}

    /**
     * The method sets second number
     * @param second second number
     */
	public void setSecond(double second) {
		this.second = second;
	}
	
	@Override
	public String toString()
	{
		return first+","+second;
	}
	
	@Override
	public boolean equals(Object o) {
	    if( !(o instanceof PrimitivePair) )
	        return false;
	    PrimitivePair p = (PrimitivePair)o;
	    return p.first == first && p.second == second;
	}
	
	@Override
	public int hashCode() {
	    return (17 + MLUtils.doubleHash(first))*31 + MLUtils.doubleHash(second); 
	}

}
