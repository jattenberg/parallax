package com.dsi.parallax.ml.util;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Random;

/**
 * not a bunch of random utilities, but rather utilities for deailing with
 * random #'s
 * 
 * @author jattenberg
 * 
 */
public class RandomUtils extends Random {

	private static final long serialVersionUID = -5846037487257657636L;
	public static final RandomUtils INSTANCE = new RandomUtils();

	public RandomUtils() {
		super();
	}

	public RandomUtils(int seed) {
		super(seed);
	}

	public synchronized double nextUniform() {
		long l = ((long) (next(26)) << 27) + next(27);
		return l / (double) (1L << 53);
	}

	/**
	 * Return random integer from Poission with parameter lambda. The mean of
	 * this distribution is lambda. The variance is lambda.
	 */
	public synchronized int nextPoisson(double lambda) {
		int v = -1;
		double l = Math.exp(-lambda), p;
		p = 1.0;
		while (p >= l) {
			p *= nextUniform();
			v++;
		}
		return v;
	}

	/** Return nextPoisson(1). */
	public synchronized int nextPoisson() {
		return nextPoisson(1);
	}

	/** Return a random boolean, equally likely to be true or false. */
	public synchronized boolean nextBoolean() {
		return (next(32) & 1 << 15) != 0;
	}

	/** Return a random boolean, with probability p of being true. */
	public synchronized boolean nextBoolean(double p) {
		double u = nextUniform();
		if (u < p)
			return true;
		return false;
	}

	/**
	 * Return a random double in the range a to b, inclusive, uniformly sampled
	 * from that range. The mean of this distribution is (b-a)/2. The variance
	 * is (b-a)^2/12
	 */
	public synchronized double nextUniform(double a, double b) {
		return a + (b - a) * nextUniform();
	}

	public synchronized int nextInt(int min, int max) {
		checkArgument(max > min, "max must be > min, given: min (%s), max (%s)", min, max);
		return nextInt(max - min) + min;
	}
	
	/** Draw a single sample from multinomial "a". */
	public synchronized int nextDiscrete(double[] a) {
		double b = 0, r = nextUniform();
		for (int i = 0; i < a.length; i++) {
			b += a[i];
			if (b > r) {
				return i;
			}
		}
		return a.length - 1;
	}

	/**
	 * draw a single sample from (unnormalized) multinomial "a", with
	 * normalizing factor "sum".
	 */
	public synchronized int nextDiscrete(double[] a, double sum) {
		double b = 0, r = nextUniform() * sum;
		for (int i = 0; i < a.length; i++) {
			b += a[i];
			if (b > r) {
				return i;
			}
		}
		return a.length - 1;
	}

	private double nextGaussian;
	private boolean haveNextGaussian = false;

	/**
	 * Return a random double drawn from a Gaussian distribution with mean 0 and
	 * variance 1.
	 */
	public synchronized double nextGaussian() {
		if (!haveNextGaussian) {
			double v1 = nextUniform(), v2 = nextUniform();
			double x1, x2;
			x1 = Math.sqrt(-2 * Math.log(v1)) * Math.cos(2 * Math.PI * v2);
			x2 = Math.sqrt(-2 * Math.log(v1)) * Math.sin(2 * Math.PI * v2);
			nextGaussian = x2;
			haveNextGaussian = true;
			return x1;
		} else {
			haveNextGaussian = false;
			return nextGaussian;
		}
	}

	/**
	 * Return a random double drawn from a Gaussian distribution with mean m and
	 * variance s2.
	 */
	public synchronized double nextGaussian(double m, double s2) {
		return nextGaussian() * Math.sqrt(s2) + m;
	}

	// generate Gamma(1,1)
	// E(X)=1 ; Var(X)=1
	/**
	 * Return a random double drawn from a Gamma distribution with mean 1.0 and
	 * variance 1.0.
	 */
	public synchronized double nextGamma() {
		return nextGamma(1, 1, 0);
	}

	/**
	 * Return a random double drawn from a Gamma distribution with mean alpha
	 * and variance 1.0.
	 */
	public synchronized double nextGamma(double alpha) {
		return nextGamma(alpha, 1, 0);
	}

	/**
	 * Return a random double drawn from a Gamma distribution with mean
	 * alpha*beta and variance alpha*beta^2.
	 */
	public synchronized double nextGamma(double alpha, double beta) {
		return nextGamma(alpha, beta, 0);
	}

	/**
	 * Return a random double drawn from a Gamma distribution with mean
	 * alpha*beta+lamba and variance alpha*beta^2. Note that this means the pdf
	 * is:
	 * <code>frac{ x^{alpha-1} exp(-x/beta) }{ beta^alpha Gamma(alpha) }</code>
	 * in other words, beta is a "scale" parameter. An alternative
	 * parameterization would use 1/beta, the "rate" parameter.
	 */
	public synchronized double nextGamma(double alpha, double beta,
			double lambda) {
		double gamma = 0;
		checkArgument(alpha > 0, "alpha must be positive, given: %s", alpha);
		checkArgument(beta > 0, "beta must be positive, given: %s", beta);

		if (alpha < 1) {
			double b, p;
			boolean flag = false;

			b = 1 + alpha * Math.exp(-1);

			while (!flag) {
				p = b * nextUniform();
				if (p > 1) {
					gamma = -Math.log((b - p) / alpha);
					if (nextUniform() <= Math.pow(gamma, alpha - 1)) {
						flag = true;
					}
				} else {
					gamma = Math.pow(p, 1.0 / alpha);
					if (nextUniform() <= Math.exp(-gamma)) {
						flag = true;
					}
				}
			}
		} else if (alpha == 1) {
			// Gamma(1) is equivalent to Exponential(1). We can
			// sample from an exponential by inverting the CDF:

			gamma = -Math.log(nextUniform());

			// There is no known closed form for Gamma(alpha != 1)...
		} else {

			// This is Best's algorithm: see pg 410 of
			// Luc Devroye's "non-uniform random variate generation"
			// This algorithm is constant time for alpha > 1.

			double b = alpha - 1;
			double c = 3 * alpha - 0.75;

			double u, v;
			double w, y, z;

			boolean accept = false;

			while (!accept) {
				u = nextUniform();
				v = nextUniform();

				w = u * (1 - u);
				y = Math.sqrt(c / w) * (u - 0.5);
				gamma = b + y;

				if (gamma >= 0.0) {
					z = 64 * w * w * w * v * v; // ie: 64 * w^3 v^2

					accept = z <= 1.0 - ((2 * y * y) / gamma);

					if (!accept) {
						accept = (Math.log(z) <= 2 * (b * Math.log(gamma / b) - y));
					}
				}
			}
		}
		return beta * gamma + lambda;
	}

	/**
	 * Return a random double drawn from an Exponential distribution with mean 1
	 * and variance 1.
	 */
	public synchronized double nextExp() {
		return nextGamma(1, 1, 0);
	}

	/**
	 * Return a random double drawn from an Exponential distribution with mean
	 * beta and variance beta^2.
	 */
	public synchronized double nextExp(double beta) {
		return nextGamma(1, beta, 0);
	}

	/**
	 * Return a random double drawn from an Exponential distribution with mean
	 * beta+lambda and variance beta^2.
	 */
	public synchronized double nextExp(double beta, double lambda) {
		return nextGamma(1, beta, lambda);
	}

	/**
	 * Return a random double drawn from an Chi-squarted distribution with mean
	 * 1 and variance 2. Equivalent to nextChiSq(1)
	 */
	public synchronized double nextChiSq() {
		return nextGamma(0.5, 2, 0);
	}

	/**
	 * Return a random double drawn from an Chi-squared distribution with mean
	 * df and variance 2*df.
	 */
	public synchronized double nextChiSq(int df) {
		return nextGamma(0.5 * (double) df, 2, 0);
	}

	/**
	 * Return a random double drawn from an Chi-squared distribution with mean
	 * df+lambda and variance 2*df.
	 */
	public synchronized double nextChiSq(int df, double lambda) {
		return nextGamma(0.5 * (double) df, 2, lambda);
	}

	/**
	 * Return a random double drawn from a Beta distribution with mean a/(a+b)
	 * and variance ab/((a+b+1)(a+b)^2).
	 */
	public synchronized double nextBeta(double alpha, double beta) {
		checkArgument(alpha > 0, "alpha must be positive, given: %s", alpha);
		checkArgument(beta > 0, "beta must be positive, given: %s", beta);
		if (alpha == 1 && beta == 1) {
			return nextUniform();
		} else if (alpha >= 1 && beta >= 1) {
			double A = alpha - 1, B = beta - 1, C = A + B, L = C * Math.log(C), mu = A
					/ C, sigma = 0.5 / Math.sqrt(C);
			double y = nextGaussian(), x = sigma * y + mu;
			while (x < 0 || x > 1) {
				y = nextGaussian();
				x = sigma * y + mu;
			}
			double u = nextUniform();
			while (Math.log(u) >= A * Math.log(x / A) + B
					* Math.log((1 - x) / B) + L + 0.5 * y * y) {
				y = nextGaussian();
				x = sigma * y + mu;
				while (x < 0 || x > 1) {
					y = nextGaussian();
					x = sigma * y + mu;
				}
				u = nextUniform();
			}
			return x;
		} else {
			double v1 = Math.pow(nextUniform(), 1 / alpha), v2 = Math.pow(
					nextUniform(), 1 / beta);
			while (v1 + v2 > 1) {
				v1 = Math.pow(nextUniform(), 1 / alpha);
				v2 = Math.pow(nextUniform(), 1 / beta);
			}
			return v1 / (v1 + v2);
		}
	}
	
	public synchronized <T extends Enum<T>> T nextEnum(T[] valueSet) {
		int index = nextInt(valueSet.length);
		return valueSet[index];
	}
}
