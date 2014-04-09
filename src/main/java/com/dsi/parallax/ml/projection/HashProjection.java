/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import com.dsi.parallax.ml.util.HashFunctionType;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;

/**
 * A random projection implementation that avoids storing the projection matrix
 * through clever hashing tricks <br>
 * <br>
 * {@link <a href="http://cseweb.ucsd.edu/~dasgupta/papers/randomf.pdf">Experiments with Random Projection</a>}
 * 
 * @author jattenberg
 */
public class HashProjection extends AbstractProjection {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;

	/** The salt used for caching */
	private final int salt;

	/** The hashcode (computed lazily) */
	private int hashcode = 0;

	/** The type of hash function used */
	private HashFunctionType hashType = HashFunctionType.JAVA;

	/**
	 * Instantiates a new hash projection.
	 * 
	 * @param in
	 *            the number of dimensions in the input space
	 * @param out
	 *            the number of dimensions of the output space
	 */
	public HashProjection(int in, int out) {
		this(in, out, 12345);
	}

	/**
	 * Instantiates a new hash projection.
	 * 
	 * @param in
	 *            the number of dimensions in the input space
	 * @param out
	 *            the number of dimensions of the output space
	 * @param salt
	 *            the salt for hashing
	 */
	public HashProjection(int in, int out, int salt) {
		super(in, out);
		this.salt = salt;
	}

	/**
	 * Project a vector in the input space into a vector belonging to hte output
	 * space using hashing
	 * 
	 * @param x
	 *            the input vector
	 * @return a vector belonging to hte output space.
	 */
	@Override
	public LinearVector project(LinearVector x) {
		LinearVector out = LinearVectorFactory.getVector(outDim);
		for (int x_i : x) {

			int x_new = hashType.hash(x_i + " " + salt, outDim);
			int psi = hashType.hash(x_i + " " + x_new + " " + salt + " psi", 2) == 1 ? 1
					: -1;
			out.updateValue(x_new, psi * x.getValue(x_i));
		}
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof HashProjection))
			return false;
		HashProjection p = (HashProjection) o;
		return (p.salt == salt && p.inDim == inDim && p.outDim == outDim);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		int result = hashcode;
		// lazy implementation
		if (result == 0) {
			result = 17;
			result = 31 * result + inDim;
			result = 31 * result * outDim;
			result = 31 * result * salt;
			hashcode = result;
		}
		return result;
	}
}
