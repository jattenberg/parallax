/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import cern.jet.stat.Gamma;

public class Stat {

    private Stat() {
    }

    /** Some constants */
    protected static final double MACHEP =  1.11022302462515654042E-16;
    protected static final double MAXLOG =  7.09782712893383996732E2;
    protected static final double MINLOG = -7.451332191019412076235E2;
    protected static final double MAXGAM = 171.624376956302725;
    protected static final double SQTPI  =  2.50662827463100050242E0;
    protected static final double SQRTH  =  7.07106781186547524401E-1;
    protected static final double LOGPI  =  1.14472988584940017414;
    
    protected static final double big    =  4.503599627370496e15;
    protected static final double biginv =  2.22044604925031308085e-16;

    /**
     * Computes the value of the hypergeometric distribution for the passed
     * values. To avoid overflows when computing the factorials, we use a small
     * trick by taking initially the logarithms and then returning the exponent.
     * 
     * log(N!) = log (1*2*...*N = log1 + log2 + .... + log N = Sum(log(i),
     * i=1..N);
     * 
     * @param D
     *            Size of database
     * @param gt
     *            Token degree
     * @param S
     *            Sample size
     * @return The value of the hypergeometric
     */
    public static double hypergeometric(long D, long gt, long S) {

        double dgt = 0;
        for (int i = 1; i <= D - gt; i++) {
            dgt += Math.log(i);
        }

        double ds = 0;
        for (int i = 1; i <= D - S; i++) {
            ds += Math.log(i);
        }

        double dgts = 0;
        for (int i = 1; i <= D - gt - S; i++) {
            dgts += Math.log(i);
        }

        double d = 0;
        for (int i = 1; i <= D; i++) {
            d += Math.log(i);
        }

        double P = Math.exp(dgt + ds - dgts - d);

        return P;
    }

    /**
     * Computes the value of the hypergeometric distribution for the passed
     * values. To avoid overflows when computing the factorials, we use a small
     * trick by taking initially the logarithms and then returning the exponent.
     * 
     * log(N!) = log (1*2*...*N = log1 + log2 + .... + log N = Sum(log(i),
     * i=1..N);
     * 
     * @param D
     *            Size of database
     * @param gt
     *            Token degree
     * @param S
     *            Sample size
     * @return The value of the hypergeometric
     */
    public static double hgapprox(long D, long gt, int S) {

        double dgt = logNfact(D - gt);

        double ds = logNfact(D - S);

        double dgts = logNfact(D - gt - S);

        double d = logNfact(D);

        double P = Math.exp(dgt + ds - dgts - d);

        return P;
    }

    public static double Beta_CDF(double x, int a, int b) {
        return Ix(x, a, b);
    }

    public static double incompleteBeta(double x, int a, int b) {
        return Beta(a, b) * Ix(x, a, b);
    }

    public static double Beta(int a, int b) {
        return Math.exp(logNfactExact(a - 1) + logNfactExact(b - 1)
                - logNfactExact(a + b - 1));
    }

    public static double Ix(double x, int a, int b) {

        double result = 0;
        for (int j = a; j <= a + b - 1; j++) {
            double m = Math.exp(logNfactExact(a + b - 1) - logNfactExact(j)
                    - logNfactExact(a + b - 1 - j));
            double n = Math.pow(x, j) * Math.pow(1 - x, a + b - 1 - j);
            result += m * n;
        }
        return result;

    }

    /**
     * Computing log(n!) using Stirling's approximation of n!
     */
    public static double logNfact(double n) {

        if (n <= 0)
            return 0;

        if (n < 100)
            return logNfactExact(n);

        // Stirling's approximation:
        // double P = Math.log(2*Math.PI*n)/2 + n*Math.log(n/Math.E);

        // Gosper's approximation
        double P = Math.log((2 * n + 1.0 / 3) * Math.PI) / 2 + n
                * Math.log(n / Math.E);

        return P;
    }

    /**
     * Computing log(n!) using Stirling's approximation of n!
     */
    private static double logNfactExact(double n) {

        if (n <= 0)
            return 0;

        double s = 0;
        for (int i = 1; i <= n; i++) {
            s += Math.log(i);
        }

        return s;
    }

    /**
     * Computing log(n!) using Stirling's approximation of n!
     */
    public static long NfactExact(long n) {

        if (n <= 0)
            return 0;

        long s = 0;
        for (int i = 1; i <= n; i++) {
            s *= i;
        }

        return s;
    }

    /**
     * Computing log(n!) using Stirling's approximation of n!
     */
    public static double binom(int n, int i) {

        return 1.0 * NfactExact(n) / (NfactExact(i) * NfactExact(n - i));

    }

    /**
     * Returns chi-squared probability for given value and degrees of freedom.
     * (The probability that the chi-squared variate will be greater than x for
     * the given degrees of freedom.)
     * 
     * @param x
     *            the value
     * @param v
     *            the number of degrees of freedom
     * @return the chi-squared probability
     */
    public static double chiSquaredProbability(double x, double v) {

        if (x < 0.0 || v < 1.0)
            return 0.0;
        return incompleteGammaComplement(v / 2.0, x / 2.0);
    }

    /**
     * Returns the Complemented Incomplete Gamma function.
     * 
     * @param a
     *            the parameter of the gamma distribution.
     * @param x
     *            the integration start point.
     */
    public static double incompleteGammaComplement(double a, double x) {

        double ans, ax, c, yc, r, t, y, z;
        double pk, pkm1, pkm2, qk, qkm1, qkm2;

        if (x <= 0 || a <= 0)
            return 1.0;

        if (x < 1.0 || x < a)
            return 1.0 - incompleteGamma(a, x);

        ax = a * Math.log(x) - x - Gamma.logGamma(a);
        if (ax < -MAXLOG)
            return 0.0;

        ax = Math.exp(ax);

        /* continued fraction */
        y = 1.0 - a;
        z = x + y + 1.0;
        c = 0.0;
        pkm2 = 1.0;
        qkm2 = x;
        pkm1 = x + 1.0;
        qkm1 = z * x;
        ans = pkm1 / qkm1;

        do {
            c += 1.0;
            y += 1.0;
            z += 2.0;
            yc = y * c;
            pk = pkm1 * z - pkm2 * yc;
            qk = qkm1 * z - qkm2 * yc;
            if (qk != 0) {
                r = pk / qk;
                t = Math.abs((ans - r) / r);
                ans = r;
            } else
                t = 1.0;

            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            if (Math.abs(pk) > big) {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
        } while (t > MACHEP);

        return ans * ax;
    }

    /**
     * Returns the Incomplete Gamma function.
     * 
     * @param a
     *            the parameter of the gamma distribution.
     * @param x
     *            the integration end point.
     */
    public static double incompleteGamma(double a, double x) {

        double ans, ax, c, r;

        if (x <= 0 || a <= 0)
            return 0.0;

        if (x > 1.0 && x > a)
            return 1.0 - incompleteGammaComplement(a, x);

        /* Compute x**a * exp(-x) / gamma(a) */
        ax = a * Math.log(x) - x - Gamma.logGamma(a);
        if (ax < -MAXLOG)
            return (0.0);

        ax = Math.exp(ax);

        /* power series */
        r = a;
        c = 1.0;
        ans = 1.0;

        do {
            r += 1.0;
            c *= x / r;
            ans += c;
        } while (c / ans > MACHEP);

        return (ans * ax / a);
    }

    public static void main(String[] args) {
        // Testing IncompleteBeta

        /*
         * for (int a =1 ; a<10; a++) for (int b =1 ; b<10; b++)
         * System.out.println("a="+a+" b="+b+" Beta(a,b)="+Beta(a,b));
         */

        double x = 0.5;
        for (int a = 0; a <= 20; a++)
            for (int b = 0; b <= 20; b++)
                System.out.println("x=" + x + " pos=" + a + " neg=" + b
                        + " Beta_CDF(x;pos,neg)=" + Beta_CDF(x, a + 1, b + 1));

    }

}
